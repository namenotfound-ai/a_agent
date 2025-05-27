
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import sys
import glob
import json
import re
import time
import uuid
import gc
import subprocess
import signal
import atexit
import socket
import tempfile

# Attempt to import PyYAML for llama-server config
try:
    import yaml
    PYYAML_AVAILABLE = True
except ImportError:
    PYYAML_AVAILABLE = False
    print("CRITICAL WARNING: PyYAML library not found. Please install it (`pip install PyYAML`) to enable YAML configuration for llama-server instances.")
    print("                 Tool server startup using --models-yml will fail without PyYAML.")
    # Exit if PyYAML is essential and missing early
    sys.exit("PyYAML is missing. Please install it with 'pip install PyYAML'.")


# Attempt to import gRPC, generate stubs if missing
try:
    import grpc
    # These will be generated if missing
    import llm_tool_service_pb2
    import llm_tool_service_pb2_grpc
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    print("WARNING: gRPC libraries or generated stubs not found. Attempting to generate stubs...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        proto_dir = os.path.join(script_dir, "proto")
        proto_file = os.path.join(proto_dir, "llm_tool_service.proto")

        if not os.path.exists(proto_dir):
            os.makedirs(proto_dir)
            print(f"Created directory: {proto_dir}")

        if not os.path.exists(proto_file):
            proto_content = """syntax = "proto3";

package llm_tool_service;

// Service definition for an LLM acting as a callable tool
service LlmTool {
  // Processes a query intended for a specialized LLM tool
  rpc ProcessToolQuery (ToolQueryRequest) returns (ToolQueryResponse);
}

message ToolQueryRequest {
  string query_text = 1;         // The actual query/prompt for the tool model
  int32 max_new_tokens = 2;     // Max tokens the tool model should generate for its response
  // Optional: Add other parameters like temperature if needed by tools
  // float temperature = 3;
}

message ToolQueryResponse {
  string response_text = 1;    // The tool model's response
  bool error = 2;              // True if an error occurred during tool processing
  string error_message = 3;    // Error details if an error occurred
}
"""
            with open(proto_file, "w") as f:
                f.write(proto_content)
            print(f"Created proto file: {proto_file}")

        protoc_command = [
            sys.executable,
            "-m", "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"--python_out={script_dir}",
            f"--grpc_python_out={script_dir}",
            proto_file
        ]
        print(f"Running gRPC stub generation: {' '.join(protoc_command)}")
        process = subprocess.run(protoc_command, capture_output=True, text=True, check=True, cwd=script_dir)
        print("gRPC stubs generated successfully. Please re-run the script.")
        sys.exit(0) # Exit so user can re-run with stubs now available
    except FileNotFoundError:
        print("ERROR: python -m grpc_tools.protoc command not found. Is grpcio-tools installed (`pip install grpcio grpcio-tools`) and in PATH?")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to generate gRPC stubs. Output:\n{e.stdout}\n{e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during gRPC stub generation: {e}")
        sys.exit(1)

MLX_LM_AVAILABLE = False
try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load, generate as mlx_generate
    MLX_LM_AVAILABLE = True
except ImportError:
    print("Warning: mlx_lm library not found. MLX models (Flask-hosted) will not be available.")

LLAMA_CPP_AVAILABLE = False
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    print("CRITICAL WARNING: llama_cpp library not found. GGUF models (Flask-hosted) will not be available.")


# --- Configuration Constants ---
# !!! IMPORTANT: SET THIS TO THE CORRECT ABSOLUTE PATH OF YOUR LLAMA.CPP SERVER EXECUTABLE !!!
# Example: "/path/to/your/llama.cpp/build/bin/llama-server" or "C:\\Users\\YourUser\\llama.cpp\\build\\bin\\llama-server.exe"
# If it's in your system PATH, "llama-server" might work, but absolute path is recommended for reliability.
LLAMA_CPP_SERVER_EXECUTABLE = "/actual/path/to/your/llama-server" # Adjust this path

CONTEXT_DB_DIRECTORY = "./context_store"
MODELS_BASE_DIRECTORY = "./models" # Base directory for model auto-discovery (relative paths below)
DEFAULT_SUMMARIZER_MODEL_NAME = "Intent"
N_GPU_LAYERS = -1 # For llama.cpp server and direct loading (-1 for all, 0 for CPU)
SERVER_PORT = 5035 # Main Flask app port
FIXED_TEMPERATURE = 0.7 # Default temperature for LLM generation

# Orchestrator Loop Parameters
MAX_ORCHESTRATOR_ITERATIONS = 5 # Max iterations for the /ask endpoint loop
MAX_SUB_QUERY_ITERATIONS = 3    # Max rounds of peer consultation within generate_with_llm_instance
MAX_JSON_PARSE_FAILURES = 3     # Max consecutive JSON parsing failures for the orchestrator

# Tool Calling Parameters
DEFAULT_SUB_QUERY_TIMEOUT_MS = 25000 # Timeout for gRPC tool calls
SUB_QUERY_MAX_RESPONSE_TOKENS = 512  # Max tokens for gRPC tool responses
SUB_QUERY_REGEX = re.compile( # Regex for SUB_QUERY tags (used by Orchestrator's internal specialists)
    r'\[SUB_QUERY\s+target_model_name="([^"]+)"'
    r'(?:\s+timeout_ms="(\d+)")?'
    r'(?:\s+max_tokens="(\d+)")?'
    r'\]'
    r'(.*?)'
    r'\[/SUB_QUERY\]',
    re.DOTALL | re.IGNORECASE
)

# Global list to keep track of managed subprocesses for cleanup
MANAGED_SUBPROCESSES = []

# --- Model Definition Class ---
class ModelDefinition:
    def __init__(self, name, model_id_or_path, type_config, n_ctx, role_description,
                 loading_strategy="always_loaded", is_orchestrator=False,
                 grpc_host="localhost", grpc_port=None,
                 is_tool_server_candidate=False):
        self.name = name
        self.model_id_or_path = model_id_or_path # This can be a filename (gguf), dir name (mlx), or absolute path
        self.type_config = type_config.lower() # "gguf" or "mlx"
        self.n_ctx = n_ctx
        self.role_description = role_description
        self.loading_strategy = loading_strategy
        self.is_orchestrator = is_orchestrator
        self.actual_path = None # Will be resolved during startup
        self.instance = None # Stores the loaded Llama instance or MLX tuple (for Flask-hosted generation)

        # gRPC specific (for when this model acts as a callable tool via its own llama-server)
        self.is_tool_server_candidate = is_tool_server_candidate and (self.type_config == "gguf") # Only GGUF can be llama-server tools
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port # Can be None for dynamic assignment for gRPC service
        self.grpc_server_process = None # Stores the subprocess object if launched
        self.temp_config_file_path = None # Path to the temporary YAML config for this server

MODEL_INVENTORY = [
    ModelDefinition(
        name="GeneralThinking", model_id_or_path="OrchestratorThinking.gguf", type_config="gguf", n_ctx=12000,
        role_description="Master orchestrator and general reasoning specialist. Manages complex tasks by delegating to other specialists and synthesizing their inputs. Can also perform high-level reasoning.",
        loading_strategy="always_loaded", is_orchestrator=True,
        is_tool_server_candidate=True
    ),
    ModelDefinition(
        name="Coder", model_id_or_path="Coder.gguf", type_config="gguf", n_ctx=8000,
        role_description="Code generation, analysis, debugging, and programming task specialist. Writes and refines code in various languages.",
        loading_strategy="always_loaded", is_tool_server_candidate=True
    ),
    ModelDefinition(
        name="FastCoder", model_id_or_path="FastCoder.gguf", type_config="gguf", n_ctx=8192,
        role_description="Rapid code generation and quick programming solutions specialist. Ideal for scaffolding, snippets, and fast iterations.",
        loading_strategy="on_demand", is_tool_server_candidate=True
    ),
    ModelDefinition(
        name="Planner", model_id_or_path="Planner.gguf", type_config="gguf", n_ctx=12000,
        role_description="Strategic planning, project management, and task breakdown specialist. Devises plans and sequences for complex operations.",
        loading_strategy="on_demand", is_tool_server_candidate=True
    ),
    ModelDefinition(
        name="Intent", model_id_or_path="Intent.gguf", type_config="gguf", n_ctx=8192,
        role_description="Intent analysis, quick summarization, and user requirement understanding. Clarifies goals and extracts key information.",
        loading_strategy="always_loaded", is_tool_server_candidate=True
    ),
    ModelDefinition(
        name="EnrichData", model_id_or_path="EnrichData.gguf", type_config="gguf", n_ctx=8192,
        role_description="Data enhancement, context addition, and information augmentation specialist. Finds and integrates relevant data.",
        loading_strategy="on_demand", is_tool_server_candidate=True
    ),
    ModelDefinition(
        name="PerformAction", model_id_or_path="PerformAction.gguf", type_config="gguf", n_ctx=12192,
        role_description="Action execution, implementation planning, and practical task completion. Focuses on turning plans into reality.",
        loading_strategy="on_demand", is_tool_server_candidate=True
    ),
    ModelDefinition(
        name="DetailOperation", model_id_or_path="DetailOperation.gguf", type_config="gguf", n_ctx=8000,
        role_description="Detailed analysis, thorough examination, and comprehensive operation handling. Examines specifics and edge cases.",
        loading_strategy="on_demand", is_tool_server_candidate=True
    ),
    # Example for an MLX model definition (uncomment and adjust path if you have one)
    # ModelDefinition(
    #     name="MistralTinyMLX", model_id_or_path="Mistral-7B-Instruct-v0.2-MLX", type_config="mlx", n_ctx=4096,
    #     role_description="A small, fast MLX model for quick responses and summarization tasks.",
    #     loading_strategy="on_demand", is_orchestrator=False, is_tool_server_candidate=False
    # )
]

app = Flask(__name__)
CORS(app)

# --- Utility Functions ---
def find_free_port():
    """Finds an available network port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def cleanup_subprocesses():
    """Terminates all managed subprocesses (llama-server instances) on script exit."""
    print("\nShutting down managed subprocesses...")
    for p_info in MANAGED_SUBPROCESSES:
        process = p_info["process"]
        port = p_info["port"]
        temp_config_file = p_info.get("temp_config_file_path")
        model_name = p_info.get("name", "Unknown Model")

        print(f"Terminating server for {model_name} (gRPC port {port}, PID: {process.pid})...")
        if process.poll() is None: # Still running
            process.terminate()
            try:
                process.wait(timeout=5)
                print(f"Server for {model_name} terminated.")
            except subprocess.TimeoutExpired:
                print(f"Server for {model_name} did not terminate gracefully, killing...")
                process.kill()
                process.wait()
                print(f"Server for {model_name} killed.")
            except Exception as e:
                print(f"Exception during subprocess wait/kill for {model_name}: {e}")
        else:
            print(f"Server for {model_name} was already terminated (exit code {process.poll()}).")

        if temp_config_file and os.path.exists(temp_config_file):
            try:
                os.remove(temp_config_file)
                print(f"Removed temporary config file: {temp_config_file}")
            except Exception as e:
                print(f"Error removing temporary config file {temp_config_file}: {e}")
    MANAGED_SUBPROCESSES.clear()

atexit.register(cleanup_subprocesses)

def start_llama_cpp_tool_server(model_def: ModelDefinition):
    """
    Starts a llama.cpp server as a gRPC tool. Includes health checks for readiness.
    """
    if not PYYAML_AVAILABLE:
        print(f"ERROR: Cannot start tool server for {model_def.name}. PyYAML is not installed.")
        return False
        
    if not model_def.is_tool_server_candidate or model_def.type_config != "gguf":
        print(f"Skipping tool server for {model_def.name}: Not a GGUF tool candidate.")
        return False

    if not model_def.actual_path or not os.path.exists(model_def.actual_path):
        print(f"ERROR: Cannot start tool server for {model_def.name}. Model file not found: {model_def.actual_path}")
        return False

    if model_def.grpc_server_process and model_def.grpc_server_process.poll() is None:
        print(f"Tool server for {model_def.name} already running (gRPC port {model_def.grpc_port}).")
        return True # Already running

    grpc_port_to_use = model_def.grpc_port if model_def.grpc_port else find_free_port()
    model_def.grpc_port = grpc_port_to_use
    http_port_for_server = find_free_port()

    server_config_data = {
        "models": [
            {
                "model_alias": model_def.name,
                "model": model_def.actual_path,
                "n_ctx": model_def.n_ctx,
                "n_gpu_layers": N_GPU_LAYERS,
            }
        ]
    }

    temp_yaml_file_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(server_config_data, f, sort_keys=False)
            temp_yaml_file_path = f.name
        model_def.temp_config_file_path = temp_yaml_file_path
        
        cmd = [
            LLAMA_CPP_SERVER_EXECUTABLE,
            "--models-yml", temp_yaml_file_path,
            "--host", model_def.grpc_host,
            "--port", str(http_port_for_server),
            "--grpc-port", str(grpc_port_to_use),
        ]

        print(f"Attempting to start tool server for {model_def.name} (HTTP: {http_port_for_server}, gRPC: {grpc_port_to_use})...")
        print(f"Command: {' '.join(cmd)}")

        preexec_fn = os.setpgrp if sys.platform != "win32" else None
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True, preexec_fn=preexec_fn)
        
        model_def.grpc_server_process = process
        MANAGED_SUBPROCESSES.append({
            "name": model_def.name,
            "process": process,
            "port": grpc_port_to_use,
            "http_port": http_port_for_server,
            "temp_config_file_path": temp_yaml_file_path
        })

        # Health check loop for server readiness
        ready_timeout_seconds = 60 # Total time to wait for server to become ready
        check_interval_seconds = 2 # How often to check
        start_check_time = time.time()
        server_ready = False
        
        print(f"Waiting for gRPC tool server for {model_def.name} to become ready (max {ready_timeout_seconds}s)...")

        while time.time() - start_check_time < ready_timeout_seconds:
            if process.poll() is not None: # Server crashed or exited
                output = ""
                try: output = process.stdout.read() # Read any remaining output
                except Exception: pass # Ignore if stream is closed
                print(f"ERROR: Tool server for {model_def.name} failed to start or exited prematurely (exit code: {process.returncode}).")
                print(f"Server Output:\n{output}")
                
                MANAGED_SUBPROCESSES[:] = [p for p in MANAGED_SUBPROCESSES if p["process"] != process]
                if temp_yaml_file_path and os.path.exists(temp_yaml_file_path):
                    try: os.remove(temp_yaml_file_path)
                    except Exception as e: print(f"Error removing temp config file {temp_yaml_file_path}: {e}")
                model_def.grpc_server_process = None
                model_def.temp_config_file_path = None
                return False
                
            try:
                address = f"{model_def.grpc_host}:{grpc_port_to_use}"
                with grpc.insecure_channel(address) as channel:
                    grpc.channel_ready_future(channel).result(timeout=check_interval_seconds - 0.5)
                print(f"gRPC tool server for {model_def.name} is READY on {address}.")
                server_ready = True
                break
            except grpc.FutureTimeoutError:
                print(f"  Still waiting for {model_def.name} on {address}...")
                time.sleep(check_interval_seconds)
            except Exception as e:
                print(f"  Connection error during health check for {model_def.name}: {e}")
                time.sleep(check_interval_seconds)

        if not server_ready:
            print(f"ERROR: gRPC tool server for {model_def.name} did not become ready within {ready_timeout_seconds}s. Terminating process.")
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill()
            
            MANAGED_SUBPROCESSES[:] = [p for p in MANAGED_SUBPROCESSES if p["process"] != process]
            if temp_yaml_file_path and os.path.exists(temp_yaml_file_path):
                try: os.remove(temp_yaml_file_path)
                except Exception as e: print(f"Error removing temp config file {temp_yaml_file_path}: {e}")
            model_def.grpc_server_process = None
            model_def.temp_config_file_path = None
            return False
        
        return True
        
    except FileNotFoundError:
        print(f"ERROR: LLAMA_CPP_SERVER_EXECUTABLE ('{LLAMA_CPP_SERVER_EXECUTABLE}') not found. Please check PATH or configuration.")
        if temp_yaml_file_path and os.path.exists(temp_yaml_file_path): os.remove(temp_yaml_file_path)
        return False
    except Exception as e:
        print(f"ERROR: Exception starting tool server for {model_def.name}: {e}")
        if model_def.grpc_server_process and model_def.grpc_server_process.poll() is None:
            model_def.grpc_server_process.kill()
        if temp_yaml_file_path and os.path.exists(temp_yaml_file_path): os.remove(temp_yaml_file_path)
        
        MANAGED_SUBPROCESSES[:] = [p_info for p_info in MANAGED_SUBPROCESSES if p_info.get("name") != model_def.name or p_info.get("process") != model_def.grpc_server_process]
        model_def.grpc_server_process = None
        model_def.temp_config_file_path = None
        return False


# --- gRPC Tool Calling Function ---
def call_grpc_llm_tool(target_model_name: str, query_text: str,
                       timeout_ms_str: str = None, max_tokens_str: str = None) -> str:
    """Calls a gRPC-enabled LLM tool (llama-server instance)."""
    if not GRPC_AVAILABLE:
        return "[TOOL_CALL_ERROR: gRPC libraries or stubs are not available.]"

    target_model_def = next((m for m in MODEL_INVENTORY if m.name == target_model_name), None)
    if not target_model_def:
        return f"[TOOL_CALL_ERROR: Target model '{target_model_name}' not found in inventory.]"
    if not target_model_def.is_tool_server_candidate:
        return f"[TOOL_CALL_ERROR: Target model '{target_model_name}' is not configured as a gRPC tool server candidate.]"
    if not target_model_def.grpc_port:
         return f"[TOOL_CALL_ERROR: Target model '{target_model_name}' gRPC port not assigned. Server may not have started.]"
    
    if not target_model_def.grpc_server_process or target_model_def.grpc_server_process.poll() is not None:
        print(f"INFO: gRPC tool server for '{target_model_name}' (gRPC port {target_model_def.grpc_port}) found stopped or not started. Attempting to start/restart...")
        if not start_llama_cpp_tool_server(target_model_def):
            return f"[TOOL_CALL_ERROR: Failed to start/restart gRPC tool server for '{target_model_name}'. Check server logs.]"
        time.sleep(1) # Brief pause after a potential restart to let the server re-initialize if it truly just launched

    host = target_model_def.grpc_host
    port = target_model_def.grpc_port
    address = f"{host}:{port}"
    timeout_seconds = (int(timeout_ms_str) / 1000.0) if timeout_ms_str and timeout_ms_str.isdigit() else (DEFAULT_SUB_QUERY_TIMEOUT_MS / 1000.0)
    tool_max_new_tokens = int(max_tokens_str) if max_tokens_str and max_tokens_str.isdigit() else SUB_QUERY_MAX_RESPONSE_TOKENS

    try:
        with grpc.insecure_channel(address) as channel:
            try:
                grpc.channel_ready_future(channel).result(timeout=max(2.0, timeout_seconds / 10.0))
            except grpc.FutureTimeoutError:
                 return f"[TOOL_CALL_ERROR: gRPC channel to '{target_model_name}' at {address} not ready. Server might be slow, unresponsive, or not serving on gRPC port.]"

            stub = llm_tool_service_pb2_grpc.LlmToolStub(channel)
            request_payload = llm_tool_service_pb2.ToolQueryRequest(
                query_text=query_text,
                max_new_tokens=tool_max_new_tokens
            )
            response = stub.ProcessToolQuery(request_payload, timeout=timeout_seconds)

            if response.error:
                return f"[TOOL_CALL_ERROR: Model '{target_model_name}' reported error: {response.error_message}]"
            return response.response_text
    except grpc.RpcError as e:
        status_code = e.code()
        details = e.details() if hasattr(e, 'details') else str(e)
        if status_code == grpc.StatusCode.DEADLINE_EXCEEDED:
            return f"[TOOL_CALL_ERROR: Call to model '{target_model_name}' at {address} timed out after {timeout_seconds}s. Details: {details}]"
        elif status_code == grpc.StatusCode.UNAVAILABLE:
             return f"[TOOL_CALL_ERROR: Model '{target_model_name}' gRPC server unavailable at {address}. Is it running, not crashing, and configured for gRPC? Details: {details}]"
        return f"[TOOL_CALL_ERROR: gRPC error calling model '{target_model_name}' at {address}: {status_code} - {details}]"
    except Exception as e:
        return f"[TOOL_CALL_ERROR: Unexpected exception calling model '{target_model_name}' at {address}: {str(e)}]"

# --- Model Loading and Management Functions (Flask-hosted instances) ---
def is_mlx_dir_heuristic(path_to_check):
    """Heuristically checks if a directory contains an MLX model."""
    if not os.path.isdir(path_to_check): return False
    has_model_files = any(f.endswith(".safetensors") for f in os.listdir(path_to_check))
    if not has_model_files: return False
    has_config = os.path.exists(os.path.join(path_to_check, "config.json"))
    has_tokenizer = (os.path.exists(os.path.join(path_to_check, "tokenizer.json")) or
                     (os.path.exists(os.path.join(path_to_check, "tokenizer_config.json")) and
                      os.path.exists(os.path.join(path_to_check, "tokenizer.model"))))
    return has_config and has_tokenizer

def resolve_and_load_models():
    """
    Resolves model paths for all models in MODEL_INVENTORY by searching under MODELS_BASE_DIRECTORY
    or using absolute paths. Then, loads 'always_loaded' Flask-hosted model generators.
    """
    print("--- Model Initialization: Resolving Paths & Loading Flask-Hosted Primary Generators ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_base_dir_full_path = os.path.join(script_dir, MODELS_BASE_DIRECTORY)

    if not os.path.exists(models_base_dir_full_path):
        print(f"WARNING: Models base directory '{models_base_dir_full_path}' not found. Please create it or adjust MODELS_BASE_DIRECTORY constant.")
        print(f"         Models relying on auto-discovery will likely not be found.")

    for model_def in MODEL_INVENTORY:
        effective_path = None
        if os.path.isabs(model_def.model_id_or_path):
            effective_path = os.path.normpath(model_def.model_id_or_path)
            if not os.path.exists(effective_path) and not (model_def.type_config == "mlx" and is_mlx_dir_heuristic(effective_path)):
                print(f"ERROR: Explicit absolute path for {model_def.name} not found or invalid type: {effective_path}")
                model_def.actual_path = None
                continue
            model_def.actual_path = effective_path
            print(f"Resolved absolute path for {model_def.name}: {effective_path}")
        else: # Search for model_id_or_path (filename or directory name) within models_base_dir
            search_name_lower = model_def.model_id_or_path.lower()
            found_path = None
            
            if not os.path.exists(models_base_dir_full_path):
                print(f"Skipping directory scan for {model_def.name} as base models directory '{models_base_dir_full_path}' does not exist.")
                model_def.actual_path = None
                continue

            for root, dirs, files in os.walk(models_base_dir_full_path):
                # Check for GGUF files
                if model_def.type_config == "gguf":
                    for file in files:
                        if file.lower() == search_name_lower and file.endswith(".gguf"):
                            found_path = os.path.join(root, file)
                            break
                if found_path: break
                
                # Check for MLX directories
                if not found_path and model_def.type_config == "mlx":
                    for dir_name in dirs:
                        if dir_name.lower() == search_name_lower:
                            dir_full_path = os.path.join(root, dir_name)
                            if is_mlx_dir_heuristic(dir_full_path):
                                found_path = dir_full_path
                                break
                    if found_path: break # Break from os.walk if found

            if found_path:
                model_def.actual_path = os.path.normpath(found_path)
                print(f"Resolved relative path for {model_def.name} ('{model_def.model_id_or_path}') to: {model_def.actual_path}")
            else:
                print(f"WARNING: Model file/directory for {model_def.name} with ID/Name '{model_def.model_id_or_path}' not found under '{models_base_dir_full_path}'.")
                print(f"         Its Flask-hosted generator will not be available.")
                if model_def.is_tool_server_candidate:
                     print(f"         Consequently, {model_def.name} cannot be started as a gRPC tool server either.")
                model_def.actual_path = None
                continue

        if model_def.loading_strategy == "always_loaded" and model_def.actual_path:
            print(f"Attempting to load 'always_loaded' Flask-hosted generator: {model_def.name} (Type: {model_def.type_config})")
            load_model_by_definition(model_def)

    print("--- Flask-Hosted Primary Generator Loading Phase Complete ---")

def load_model_by_definition(model_def: ModelDefinition):
    """Loads a Flask-hosted model generator instance based on its definition."""
    if model_def.instance:
        return model_def.instance

    if not model_def.actual_path or not os.path.exists(model_def.actual_path) and not (model_def.type_config == "mlx" and is_mlx_dir_heuristic(model_def.actual_path)):
        print(f"Error: Cannot load Flask-hosted generator for {model_def.name}. Path not resolved or does not exist/invalid type: {model_def.actual_path}")
        return None

    instance = None
    print(f"Loading Flask-hosted generator: {model_def.name} (Type: {model_def.type_config}, Path: {model_def.actual_path}, N_CTX: {model_def.n_ctx})")
    load_time_start = time.time()
    try:
        if model_def.type_config == "gguf":
            if not LLAMA_CPP_AVAILABLE:
                print(f"Error: llama_cpp library not available, cannot load GGUF model {model_def.name} for Flask hosting.")
                return None
            instance = Llama(model_path=model_def.actual_path, n_ctx=model_def.n_ctx, n_gpu_layers=N_GPU_LAYERS, verbose=False)
        elif model_def.type_config == "mlx":
            if not MLX_LM_AVAILABLE:
                print(f"Error: mlx_lm library not available, cannot load MLX model {model_def.name} for Flask hosting.")
                return None
            model_mlx, tokenizer_mlx = mlx_load(model_def.actual_path)
            instance = (model_mlx, tokenizer_mlx)
        else:
            print(f"ERROR: Unknown or unsupported model type '{model_def.type_config}' for {model_def.name} Flask-hosted generator. Skipping load.")
            return None

        model_def.instance = instance
        print(f"Flask-hosted generator for {model_def.name} loaded successfully in {time.time() - load_time_start:.2f} seconds.")
    except Exception as e:
        print(f"Error loading Flask-hosted generator for model {model_def.name}: {e}")
        model_def.instance = None
    return model_def.instance

def unload_model_by_name(model_name: str):
    """Unloads a Flask-hosted model generator if its loading_strategy is not 'always_loaded'."""
    model_def = next((m for m in MODEL_INVENTORY if m.name == model_name), None)
    if not model_def:
        return

    if model_def.loading_strategy == "always_loaded":
        return

    if model_def.instance:
        print(f"Unloading on-demand Flask-hosted generator for model: {model_def.name}")
        del model_def.instance
        model_def.instance = None
        gc.collect()
        if model_def.type_config == "mlx" and MLX_LM_AVAILABLE:
            try:
                mx.clear_cache()
                print(f"Cleared MLX cache after unloading {model_name}.")
            except Exception as e:
                print(f"Note: Error during mx.clear_cache() for {model_name}: {e}")

def get_model_instance_by_name(model_name: str):
    """Retrieves a Flask-hosted model instance, loading it if not already loaded and allowed."""
    model_def = next((m for m in MODEL_INVENTORY if m.name == model_name), None)
    if not model_def:
        raise ValueError(f"Model '{model_name}' not found in inventory.")

    if not model_def.instance:
        loaded_instance = load_model_by_definition(model_def)
        if not loaded_instance:
            raise RuntimeError(f"Failed to load Flask-hosted generator for '{model_name}'.")
        return loaded_instance
    return model_def.instance

# --- Core Generation Logic ---
def _perform_raw_generation_for_model(model_def: ModelDefinition, prompt: str, max_tokens: int, temperature: float, stop_tokens: list):
    """Internal helper to perform raw text generation using the model's loaded instance."""
    if not model_def.instance:
        raise RuntimeError(f"Model instance for {model_def.name} is not loaded.")

    try:
        if model_def.type_config == "gguf":
            if not LLAMA_CPP_AVAILABLE:
                return "[GENERATION_ERROR: llama_cpp library not available for GGUF models]"
            output = model_def.instance(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=list(set(stop_tokens)),
                echo=False,
                repeat_penalty=1.2
            )
            return output['choices'][0]['text'].strip() if output and output.get('choices') and output['choices'][0].get('text') else ""

        elif model_def.type_config == "mlx":
            if not MLX_LM_AVAILABLE:
                return "[GENERATION_ERROR: mlx_lm library not available for MLX models]"
            mlx_model, mlx_tokenizer = model_def.instance
            raw_output = mlx_generate(
                mlx_model, mlx_tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False
            ).strip()
            
            for stop_t in stop_tokens:
                if stop_t in raw_output:
                    raw_output = raw_output.split(stop_t, 1)[0].strip()
                    break
            return raw_output
        
        return f"[GENERATION_ERROR: Unsupported model type '{model_def.type_config}' for {model_def.name}]"
            
    except Exception as e:
        print(f"ERROR during raw generation for {model_def.name}: {e}")
        return f"[GENERATION_ERROR: {str(e)[:150]}]"


def simple_generate(prompt: str, model_def: ModelDefinition, max_tokens: int = 300, temperature: float = FIXED_TEMPERATURE):
    """Simple generation for direct queries with role awareness, using Flask-hosted instance."""
    if not model_def.instance:
        print(f"Warning: Instance for {model_def.name} not found in simple_generate. Attempting load...")
        load_model_by_definition(model_def)
        if not model_def.instance:
            raise ValueError(f"Model {model_def.name} Flask-hosted generator not loaded and could not be loaded.")
    
    if any(word in prompt.lower() for word in ["quickly", "simple", "brief", "short", "outline", "summarize"]):
        role_aware_prompt = f"You are {model_def.name}. {prompt}\n\nProvide a direct, practical response:"
    else:
        role_aware_prompt = f"You are {model_def.name}.\n"
        role_aware_prompt += f"YOUR ROLE: {model_def.role_description}\n\n"
        role_aware_prompt += f"REQUEST: {prompt}\n\n"
        role_aware_prompt += f"YOUR RESPONSE (be concise and direct):\n"
    
    stop_tokens = [
        "\n\n\n", "\nHuman:", "\nUSER:", "\n<|user|>", "<|im_end|>",
        "Related questions", "What is the difference", "Further information:",
        "How to list", "\nQ:", "\nQuestion:", "```\n\n",
        "[COLLABORATION", "[SOLUTION", "[TASK", "[CONSULT_PEER"
    ]
    
    return _perform_raw_generation_for_model(model_def, role_aware_prompt, max_tokens, temperature, stop_tokens)


def extract_peer_consultation_requests(response_text: str) -> list:
    """Extract [CONSULT_PEER model="Name" query="Details"] requests from model response"""
    consultation_pattern = re.compile(
        r'\[CONSULT_PEER\s+model="([^"]+)"(?:\s+timeout_ms="(\d+)")?(?:\s+max_tokens="(\d+)")?\s+query="((?:[^"\\]|\\.)*)"\]',
        re.IGNORECASE | re.DOTALL
    )
    consultations = []
    for match in consultation_pattern.finditer(response_text):
        query_content = match.group(4).replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')
        consultations.append({
            "target_model": match.group(1).strip(),
            "timeout_ms": match.group(2) if match.group(2) else str(DEFAULT_SUB_QUERY_TIMEOUT_MS),
            "max_tokens": match.group(3) if match.group(3) else str(SUB_QUERY_MAX_RESPONSE_TOKENS),
            "query": query_content.strip()
        })
    return consultations

def get_available_peer_models(current_model_name: str) -> list:
    """Get list of available peer models (excluding current model) that have a Flask-hosted instance or are gRPC tool candidates."""
    peers = []
    for mdef in MODEL_INVENTORY:
        if mdef.name == current_model_name:
            continue
        if mdef.instance or mdef.is_tool_server_candidate:
            peers.append(mdef.name)
    return peers


def generate_model_response(model_def: ModelDefinition, prompt: str, temperature: float, stop_tokens: list, max_tokens: int = 400):
    """Generate a response from a specific Flask-hosted model instance with the given prompt"""
    if not model_def.instance:
        print(f"Warning: Instance for {model_def.name} not found in generate_model_response. Attempting load...")
        load_model_by_definition(model_def)
        if not model_def.instance:
            error_msg = f"[GENERATION_ERROR: Model {model_def.name} Flask-hosted generator not loaded and could not be loaded.]"
            print(error_msg)
            return error_msg
            
    return _perform_raw_generation_for_model(model_def, prompt, max_tokens, temperature, stop_tokens)


def handle_peer_consultation(target_model_name: str, query: str, conversation_context: list,
                           timeout_ms_str: str = None, max_tokens_str: str = None, debug_log=None) -> str:
    if debug_log is None: debug_log = []
    
    target_model_def = next((m for m in MODEL_INVENTORY if m.name == target_model_name), None)
    if not target_model_def:
        error_msg = f"[PEER_ERROR: Target peer model '{target_model_name}' not found in inventory]"
        debug_log.append(error_msg)
        return error_msg

    if target_model_def.is_tool_server_candidate and GRPC_AVAILABLE:
        debug_log.append(f"Attempting gRPC consultation with {target_model_name} for query: \"{query[:70]}...\"")
        grpc_response = call_grpc_llm_tool(target_model_name, query, timeout_ms_str, max_tokens_str)
        
        if not grpc_response.startswith("[TOOL_CALL_ERROR"):
            debug_log.append(f"gRPC consultation with {target_model_name} successful.")
            return grpc_response
        else:
            debug_log.append(f"gRPC consultation with {target_model_name} failed: {grpc_response}. Will attempt fallback to Flask-hosted instance if available.")
    
    debug_log.append(f"Attempting direct Flask-hosted instance consultation with {target_model_name}.")
    try:
        if not target_model_def.instance:
            debug_log.append(f"Flask-hosted instance for peer {target_model_name} not loaded. Attempting to load...")
            load_model_by_definition(target_model_def)
        
        if not target_model_def.instance:
            error_msg = f"[PEER_ERROR: Could not load Flask-hosted instance for '{target_model_name}' after gRPC attempt/fallback]"
            debug_log.append(error_msg)
            return error_msg
        
        consultation_prompt = build_consultation_prompt(target_model_name, query, conversation_context, debug_log)
        
        max_tokens_int = int(max_tokens_str) if max_tokens_str and max_tokens_str.isdigit() else SUB_QUERY_MAX_RESPONSE_TOKENS
        
        result = generate_model_response(
            target_model_def,
            consultation_prompt,
            FIXED_TEMPERATURE,
            ["\nHuman:", "\nUSER:", "\n<|user|>", "[CONSULTATION_COMPLETE]"],
            max_tokens=max_tokens_int
        )
        
        if result.startswith("[GENERATION_ERROR") or result.startswith("[MODEL_RESPONSE_ERROR"):
             debug_log.append(f"Flask-hosted consultation with {target_model_name} resulted in generation error: {result}")
        else:
            debug_log.append(f"Flask-hosted consultation with {target_model_name} successful.")
        return result if result.strip() else "[PEER_ERROR: Empty response from Flask-hosted direct instance]"
        
    except Exception as e:
        error_msg = f"[PEER_ERROR: Exception during direct Flask-hosted consultation with {target_model_name}: {str(e)[:150]}]"
        debug_log.append(error_msg)
        return error_msg


def build_consultation_prompt(target_model_name: str, query: str, conversation_context: list, debug_log=None) -> str:
    if debug_log is None: debug_log = []

    target_model_def = next((m for m in MODEL_INVENTORY if m.name == target_model_name), None)
    if not target_model_def:
        debug_log.append(f"CONSULT_PROMPT_BUILD_ERROR: Model def for {target_model_name} not found.")
        return f"You are being consulted on the following query: \"{query}\". Please provide your expert input and end with [CONSULTATION_COMPLETE]"

    prompt = f"You are {target_model_def.name}, an AI specialist with expertise in: \"{target_model_def.role_description}\".\n"
    prompt += "You are being consulted by another AI specialist who requires your specific expertise to help resolve a larger user request.\n\n"
    prompt += f"THE SPECIFIC QUESTION FOR YOU (Focus ONLY on this query):\n\"\"\"\n{query}\n\"\"\"\n\n"

    if conversation_context:
        prompt += "FOR CONTEXT, here are the last few exchanges in the main conversation (most recent first):\n"
        for entry in reversed(conversation_context[-3:]):
            model_name_entry = entry.get('model', 'UnknownModel')
            response_entry = entry.get('response', 'No response recorded.')
            role_desc_entry = entry.get('role_description', 'N/A')
            
            if entry.get("role") == "peer_consultation":
                 prompt += f"  - Prior consultation result from {model_name_entry} (queried by {entry.get('consulted_by', 'N/A')} about \"{entry.get('consultation_query', 'N/A')[:60]}...\"): {response_entry[:150]}...\n---\n"
            else:
                 prompt += f"  - Main contribution from {model_name_entry} ({role_desc_entry}): {response_entry[:150]}...\n---\n"
        prompt += "\n"
    
    prompt += "YOUR TASK:\n"
    prompt += "1. Directly address the specific question asked of you using your unique expertise.\n"
    prompt += "2. Provide a concise, actionable, and focused response.\n"
    prompt += "3. Do NOT attempt to manage the overall problem or delegate further. Simply answer the specific query.\n"
    prompt += "4. CRITICAL: Conclude your entire expert input with the exact phrase: [CONSULTATION_COMPLETE]\n\n"
    prompt += f"YOUR EXPERT RESPONSE AS {target_model_def.name} (Remember to end with [CONSULTATION_COMPLETE]):\n"
    
    return prompt

def synthesize_collaborative_response(conversation_context: list, original_request: str, debug_log=None) -> str:
    """Synthesizes a final answer from the conversation context, preferring the last main response."""
    if debug_log is None: debug_log = []

    if not conversation_context:
        msg = "No collaborative response generated due to empty conversation context."
        debug_log.append(f"SYNTHESIS_WARN: {msg}")
        return msg

    last_main_response_entry = None
    for i in range(len(conversation_context) - 1, -1, -1):
        entry = conversation_context[i]
        is_error_response = entry["response"].startswith(("[PEER_ERROR", "[TOOL_CALL_ERROR", "[GENERATION_ERROR"))
        if entry.get("role") != "peer_consultation" and not is_error_response:
            if is_collaboration_complete(entry["response"]) or \
               (len(entry["response"]) > 50 and not extract_peer_consultation_requests(entry["response"])):
                last_main_response_entry = entry
                debug_log.append(f"SYNTHESIS: Identified last main response from {entry['model']} (Round {entry['round']}) as candidate for final answer.")
                break
    
    if not last_main_response_entry:
        debug_log.append(f"SYNTHESIS_WARN: No clear 'main' response found. Looking for any valid peer consultation.")
        for i in range(len(conversation_context) - 1, -1, -1):
            entry = conversation_context[i]
            is_error_response = entry["response"].startswith(("[PEER_ERROR", "[TOOL_CALL_ERROR", "[GENERATION_ERROR"))
            if entry.get("role") == "peer_consultation" and not is_error_response and entry["response"].strip():
                last_main_response_entry = entry
                debug_log.append(f"SYNTHESIS: Using last valid peer response from {entry['model']} as fallback final answer.")
                break
    
    if not last_main_response_entry:
        msg = "No conclusive response found in collaboration after checking main and peer contributions."
        debug_log.append(f"SYNTHESIS_ERROR: {msg}")
        valid_responses = [e["response"] for e in conversation_context if e["response"].strip() and not e["response"].startswith(("[PEER_ERROR", "[TOOL_CALL_ERROR", "[GENERATION_ERROR"))]
        if valid_responses:
            final_text = "\n---\n".join(valid_responses[-2:])
            return f"Collaborative effort result (summary of last valid exchanges):\n{final_text}"
        return msg


    final_text = last_main_response_entry["response"]

    markup_to_clean = [
        "[COLLABORATION_COMPLETE", "[SOLUTION_COMPLETE", "[TASK_COMPLETE",
        "[CONSULT_PEER", "[CONSULTATION_COMPLETE]"
    ]
    for marker in markup_to_clean:
        parts = re.split(re.escape(marker), final_text, maxsplit=1, flags=re.IGNORECASE)
        final_text = parts[0].strip()
            
    successful_peer_consultations = [
        entry for entry in conversation_context 
        if entry.get("role") == "peer_consultation" 
        and entry["response"].strip() 
        and not entry["response"].startswith(("[PEER_ERROR", "[TOOL_CALL_ERROR", "[GENERATION_ERROR"))
    ]
    
    if len(final_text) < 100 and successful_peer_consultations and last_main_response_entry.get("role") != "peer_consultation":
        final_text += "\n\nKey inputs from specialist consultations:\n"
        for pc_entry in successful_peer_consultations:
            peer_summary_text = pc_entry["response"]
            for marker in markup_to_clean:
                parts = re.split(re.escape(marker), peer_summary_text, maxsplit=1, flags=re.IGNORECASE)
                peer_summary_text = parts[0].strip()

            if peer_summary_text:
                 final_text += f"- {pc_entry['model']} (on \"{pc_entry.get('consultation_query', 'N/A')[:40]}...\"): {peer_summary_text[:150]}...\n"
        debug_log.append(f"SYNTHESIS: Appended summaries from {len(successful_peer_consultations)} peer consultations as main response was short.")
                 
    final_result = final_text.strip()
    if not final_result:
        msg = "Collaboration resulted in an empty final response after synthesis."
        debug_log.append(f"SYNTHESIS_WARN: {msg}")
        return msg
        
    debug_log.append(f"SYNTHESIS_RESULT: '{final_result[:200]}...'")
    return final_result


def build_collaborative_prompt(initial_request: str, conversation_context: list, current_model_name: str, round_num: int, debug_log=None):
    """Builds the prompt for a model engaged in a collaborative turn."""
    if debug_log is None: debug_log = []

    model_def = next((m for m in MODEL_INVENTORY if m.name == current_model_name), None)
    if not model_def:
        debug_log.append(f"COLLAB_PROMPT_BUILD_ERROR: Model definition for {current_model_name} not found.")
        return initial_request

    available_peers_names = get_available_peer_models(current_model_name)
    peer_details_list = []
    for peer_name_iter in available_peers_names:
        peer_def_iter = next((m for m in MODEL_INVENTORY if m.name == peer_name_iter), None)
        if peer_def_iter:
            status = []
            if peer_def_iter.is_tool_server_candidate: status.append("gRPC-Tool")
            if peer_def_iter.instance: status.append("Flask-Direct")
            status_str = f" ({', '.join(status)})" if status else ""

            peer_details_list.append(f"- {peer_def_iter.name}{status_str}: Specializes in \"{peer_def_iter.role_description}\"")
    
    peer_list_str = "\n".join(peer_details_list) if peer_details_list else "No other specialists currently identified as available for consultation."

    prompt = f"You are {current_model_name}, an AI specialist with the role: \"{model_def.role_description}\".\n"
    prompt += f"This is COLLABORATION ROUND {round_num} focused on the ORIGINAL USER REQUEST.\n\n"
    prompt += f"ORIGINAL USER REQUEST: \"\"\"\n{initial_request}\n\"\"\"\n\n"

    if conversation_context:
        prompt += "CURRENT CONVERSATION THREAD (most recent exchanges first):\n"
        for entry in reversed(conversation_context[-4:]):
            entry_model = entry.get('model', 'UnknownModel')
            entry_response = entry.get('response', 'No response recorded.')
            entry_role_text = entry.get('role_description', 'N/A')
            
            if entry.get("role") == "peer_consultation":
                 prompt += f"  - Consultation result from {entry_model} (Role: {entry_role_text}, queried by {entry.get('consulted_by', 'N/A')} about \"{entry.get('consultation_query', 'N/A')[:70]}...\"): {entry_response[:200]}...\n---\n"
            else:
                 prompt += f"  - Contribution from {entry_model} (Role: {entry_role_text}): {entry_response[:200]}...\n---\n"
        prompt += "\n"

    prompt += f"YOUR CURRENT TASK (as {current_model_name} in Round {round_num}):\n"
    prompt += "1. REVIEW the Original User Request and the Conversation Thread carefully.\n"
    prompt += "2. IDENTIFY what contribution YOU can make based on YOUR specific role and expertise.\n"
    prompt += "3. If you need specific input from another specialist to make progress OR if a distinct sub-task is clearly another specialist's responsibility, you MUST delegate. Use the format: [CONSULT_PEER model=\"ModelName\" query=\"Your precise and actionable question for the peer, focused on their specialty.\"]\n"
    prompt += "   - Ensure your query for the peer is self-contained and provides enough context for them to answer effectively.\n"
    prompt += "   - You can make multiple [CONSULT_PEER] calls if different expertise is needed for different sub-parts.\n"
    prompt += "4. If, after your contribution and any necessary consultations THIS ROUND, you believe the Original User Request is fully addressed, CONCLUDE your response with the exact phrase: [COLLABORATION_COMPLETE]\n"
    prompt += "5. Otherwise, provide your expert contribution. If you delegate, DO NOT use [COLLABORATION_COMPLETE] in the same response. Let the next round integrate results.\n"
    prompt += "6. Your response should be your direct contribution. Do not just say 'I will consult X'. Provide your input, and then IF NEEDED, add the [CONSULT_PEER...] tags.\n\n"
    
    prompt += "AVAILABLE PEER SPECIALISTS FOR CONSULTATION (Do not consult yourself. Refer to their roles):\n"
    prompt += f"{peer_list_str}\n\n"

    prompt += f"YOUR RESPONSE AS {current_model_name} (Provide your contribution and then any [CONSULT_PEER] tags OR [COLLABORATION_COMPLETE] if fully resolved by your current response):\n"
    
    return prompt


def generate_with_llm_instance(initial_prompt_text: str, model_def: ModelDefinition,
                               temperature_override=None, stop_tokens_override=None, 
                               enable_peer_collaboration=True, debug_log=None):
    """
    Manages a multi-turn collaborative generation session using a single Flask-hosted LLM as the primary
    agent, allowing it to delegate to other models (peers) via [CONSULT_PEER] tags.
    """
    if debug_log is None: debug_log = []

    if not model_def.instance:
        debug_log.append(f"CRITICAL_ERROR: Model {model_def.name} instance is NOT LOADED for generate_with_llm_instance.")
        load_model_by_definition(model_def)
        if not model_def.instance:
            error_msg = f"FATAL: {model_def.name} Flask-hosted instance could not be loaded for generation."
            debug_log.append(error_msg)
            raise ValueError(error_msg)

    conversation_context = []
    current_processing_model_def = model_def
    
    base_stop_tokens = [
        "\nHuman:", "\nUSER:", "\n<|user|>", "<|im_end|>", "<|endoftext|>",
        "[COLLABORATION_COMPLETE", "[SOLUTION_COMPLETE", "[TASK_COMPLETE"
    ]
    if stop_tokens_override: base_stop_tokens.extend(stop_tokens_override)
    final_stop_tokens = list(set(base_stop_tokens))
    
    current_temp = temperature_override if temperature_override is not None else FIXED_TEMPERATURE

    for collaboration_round in range(1, MAX_SUB_QUERY_ITERATIONS + 1):
        debug_log.append(f"\n--- COLLAB ROUND {collaboration_round} --- Current Model: {current_processing_model_def.name} ---")

        active_prompt = build_collaborative_prompt(
            initial_prompt_text,
            conversation_context,
            current_processing_model_def.name,
            collaboration_round,
            debug_log
        )
        
        model_response_text = generate_model_response(
            current_processing_model_def,
            active_prompt,
            current_temp,
            final_stop_tokens,
            max_tokens=SUB_QUERY_MAX_RESPONSE_TOKENS + 200
        )

        if not model_response_text.strip() or model_response_text.startswith(("[GENERATION_ERROR", "[MODEL_RESPONSE_ERROR")):
            debug_log.append(f"Round {collaboration_round} for {current_processing_model_def.name}: Received empty or error response ('{model_response_text[:100]}...'). Ending this collaboration attempt.")
            break
        
        debug_log.append(f"Response from {current_processing_model_def.name} (Round {collaboration_round}): '{model_response_text[:250]}...'")

        current_turn_context = {
            "model": current_processing_model_def.name,
            "round": collaboration_round,
            "response": model_response_text,
            "role_description": current_processing_model_def.role_description
        }
        conversation_context.append(current_turn_context)

        if is_collaboration_complete(model_response_text):
            debug_log.append(f"{current_processing_model_def.name} indicated [COLLABORATION_COMPLETE]. Collaboration chain ends.")
            break
            
        if not enable_peer_collaboration:
             debug_log.append(f"Peer collaboration globally disabled. Ending after {current_processing_model_def.name}'s turn.")
             break

        peer_consult_requests = extract_peer_consultation_requests(model_response_text)

        if not peer_consult_requests:
            debug_log.append(f"{current_processing_model_def.name} made no peer consultation requests this round. Its response stands as the current result for this chain.")
            break
        
        debug_log.append(f"{current_processing_model_def.name} made {len(peer_consult_requests)} peer consultation request(s).")
        for consult_req in peer_consult_requests:
            target_peer_name = consult_req["target_model"]
            peer_query = consult_req["query"]
            debug_log.append(f"  Processing consultation: {current_processing_model_def.name} -> {target_peer_name}. Query: \"{peer_query[:70]}...\"")

            peer_response_text = handle_peer_consultation(
                target_peer_name,
                peer_query,
                conversation_context,
                consult_req.get("timeout_ms"),
                consult_req.get("max_tokens"),
                debug_log
            )
            
            debug_log.append(f"  Response from peer {target_peer_name}: '{peer_response_text[:150]}...'")
            
            peer_def_consulted = next((m for m in MODEL_INVENTORY if m.name == target_peer_name), None)
            
            conversation_context.append({
                "model": target_peer_name,
                "round": collaboration_round,
                "response": peer_response_text,
                "role": "peer_consultation",
                "consulted_by": current_processing_model_def.name,
                "consultation_query": peer_query,
                "role_description": peer_def_consulted.role_description if peer_def_consulted else "Peer Consultant"
            })
        
        if collaboration_round >= MAX_SUB_QUERY_ITERATIONS:
            debug_log.append(f"Max collaboration rounds ({MAX_SUB_QUERY_ITERATIONS}) reached. Moving to synthesis.")
            break
            
    final_answer = synthesize_collaborative_response(conversation_context, initial_prompt_text, debug_log)
    debug_log.append(f"FINAL SYNTHESIZED ANSWER (len {len(final_answer)}): '{final_answer[:300]}...'")
    return final_answer


def is_collaboration_complete(response_text: str) -> bool:
    """Check if the model explicitly declared collaboration complete."""
    response_lower = response_text.lower()
    completion_signals = [
        "[collaboration_complete",
        "[solution_complete",
        "[task_complete"
    ]
    for signal in completion_signals:
        if signal in response_lower:
            return True
    return False


# --- Context Database (File-based) Functions ---
def ensure_context_db_dir():
    """Ensures the context database directory exists."""
    if not os.path.exists(CONTEXT_DB_DIRECTORY):
        os.makedirs(CONTEXT_DB_DIRECTORY)

def sanitize_filename(name: str):
    """Sanitizes a string for use as a filename."""
    name = re.sub(r'[^\w\.-]', '_', name)
    return name[:150]

def store_context_in_db(unique_id: str, content_to_store: str):
    """Stores content in the context database with a unique ID."""
    ensure_context_db_dir()
    filepath = os.path.join(CONTEXT_DB_DIRECTORY, f"{sanitize_filename(unique_id)}.txt")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content_to_store)
        return True, f"Content stored successfully at ID '{unique_id}'."
    except Exception as e:
        return False, f"Error storing context to ID '{unique_id}' ({filepath}): {e}"

def retrieve_context_from_db(unique_id_or_query: str):
    """Retrieves content from the context database by ID."""
    ensure_context_db_dir()
    filepath = os.path.join(CONTEXT_DB_DIRECTORY, f"{sanitize_filename(unique_id_or_query)}.txt")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read(), f"Content retrieved successfully for ID '{unique_id_or_query}'."
        except Exception as e:
            return None, f"Error retrieving context for ID '{unique_id_or_query}' ({filepath}): {e}"
    return None, f"Context ID '{unique_id_or_query}' not found."

def summarize_text_with_model(text_to_summarize: str, target_model_name_override: str = None, summarization_instructions_override: str = None):
    """Summarizes a given text using a specified (or default) summarizer model."""
    summarizer_name = target_model_name_override or DEFAULT_SUMMARIZER_MODEL_NAME
    summarizer_def = next((m for m in MODEL_INVENTORY if m.name == summarizer_name), None)
    if not summarizer_def:
        return f"[SUMMARIZATION_ERROR: Summarizer model '{summarizer_name}' not found in inventory.]", "Error"

    debug_log_summary = [f"SUMMARY_INIT: Target '{summarizer_name}', Text len {len(text_to_summarize)}"]
    try:
        get_model_instance_by_name(summarizer_name)
        
        prompt_text = summarization_instructions_override or (
            "You are a summarization expert. Concisely summarize the following text, focusing on key facts, decisions, and outcomes. "
            "This summary will provide context for other AI agents or for project tracking. Be brief yet comprehensive."
        )
        full_prompt_for_summarizer = f"{prompt_text}\n\nTEXT TO SUMMARIZE:\n---\n{text_to_summarize}\n---\n\nYOUR CONCISE SUMMARY:"
        
        summary = generate_with_llm_instance(
            full_prompt_for_summarizer,
            summarizer_def, 
            temperature_override=0.3,
            enable_peer_collaboration=False,
            debug_log=debug_log_summary
        )
        
        debug_log_summary.append(f"SUMMARY_RESULT by {summarizer_name}: '{summary[:100]}...'")
        return summary, f"Summarized successfully by {summarizer_name}."
    except Exception as e:
        err_msg = f"[SUMMARIZATION_ERROR: Error during summarization with {summarizer_name}: {e}. Log: {debug_log_summary}]"
        return err_msg, "Error"

# --- Orchestrator Tool Manifest Builder ---
def build_orchestrator_tool_manifest():
    """Builds the manifest of available tools and instructions for the orchestrator model."""
    manifest = "You are a master orchestrator AI. Your primary goal is to understand a user's complex request, break it down into manageable sub-tasks, and then delegate these sub-tasks to specialist AI assistants. You must also manage a Context Database for persistent information. Synthesize results from specialists and the database to provide a final, comprehensive answer to the user.\n\n"
    manifest += "CRITICAL INSTRUCTIONS FOR ORCHESTRATOR:\n"
    manifest += "1.  REASONING FIRST: Before outputting JSON, perform your internal reasoning. Analyze the user request, current state, and previous tool outputs.\n"
    manifest += "2.  UPDATE STATE: Maintain a detailed 'orchestrator_internal_state' reflecting your plan, completed steps, gathered information, Context DB IDs, etc.\n"
    manifest += "3.  JSON ACTION: Your SOLE output related to actions MUST be a single, valid JSON object within ```json ... ``` delimiters. NO TEXT BEFORE OR AFTER THE JSON BLOCK.\n"
    manifest += "4.  SUB_QUERY FOR SPECIALISTS: If a specialist needs to consult *another* specialist (peer-to-peer), instruct them to use [SUB_QUERY target_model_name=\"ModelName\" query=\"...\"] within the 'task_for_specialist' you provide to them. YOU, the orchestrator, use CALL_SPECIALIST_MODEL, not SUB_QUERY.\n\n"

    manifest += "AVAILABLE TOOLS FOR ORCHESTRATOR (You call these via 'next_action' in JSON):\n\n"
    manifest += "A. SPECIALIST AI ASSISTANTS (Tool: CALL_SPECIALIST_MODEL):\n"
    manifest += "   - Purpose: Delegate specific sub-tasks to AI models with specialized roles.\n"
    for model_def in MODEL_INVENTORY:
        if not model_def.is_orchestrator:
            manifest += f"   - Name: \"{model_def.name}\"\n"
            manifest += f"     Role: {model_def.role_description}\n"
            if model_def.is_tool_server_candidate:
                 manifest += f"     (This specialist can use [SUB_QUERY] to consult other specialists if its gRPC server is active.)\n"
            else:
                 manifest += f"     (This specialist relies on Flask-hosted direct calls for any sub-queries it might make if it were enabled to do so, or for your direct call to it.)\n"


    manifest += "\nB. CONTEXT DATABASE (Context DB) TOOL:\n"
    manifest += "   - Purpose: Store, retrieve, and request summaries of information for long-term memory or to manage large data between steps.\n"
    manifest += "   - Operations & Required Parameters for 'action_parameters':\n"
    manifest += "     1. action_type: \"STORE_IN_CONTEXT_DB\"\n        Params: {\"unique_storage_id\": \"<YourDescriptiveID>\", \"content_to_store\": \"<TextOrDataToSave>\"}\n"
    manifest += "     2. action_type: \"RETRIEVE_FROM_CONTEXT_DB\"\n        Params: {\"unique_storage_id_or_query\": \"<IDToRetrieve>\"}\n"
    manifest += f"    3. action_type: \"REQUEST_SUMMARY\"\n        Params: {{\"text_to_summarize\": \"<LongText>\"}}. Optional: {{\"target_summarizer_model\": \"<NameFromList, default: {DEFAULT_SUMMARIZER_MODEL_NAME}>\", \"summarization_instructions\": \"<CustomInstructions>\"}}\n"

    manifest += "\nC. OTHER ORCHESTRATOR ACTIONS:\n"
    manifest += "   - Operations & Required Parameters for 'action_parameters':\n"
    manifest += "     1. action_type: \"SELF_GENERATE\"\n        Params: {\"generated_segment\": \"<Text you are generating directly as part of the plan or as an interim note. This gets logged but doesn't directly go to user unless part of FINAL_RESPONSE.>\"}\n"
    manifest += "     2. action_type: \"FINAL_RESPONSE\"\n        Params: {\"final_answer_content\": \"<The complete, synthesized final answer for the user. This terminates the orchestration loop.>\"}\n"
    
    manifest += "\nJSON OUTPUT STRUCTURE (This entire block MUST be your response when taking an action):\n"
    manifest += "```json\n"
    manifest += """{
  "orchestrator_reasoning": "Your detailed thought process: current understanding of the user's request, analysis of the situation and previous tool outputs, justification for the chosen action, and how it helps achieve the overall goal. This reasoning MUST precede this JSON block generation in your thought process.",
  "orchestrator_internal_state": "Your comprehensive, updated internal state: ongoing plan, completed sub-tasks, accumulated knowledge, references to Context DB IDs used or created, intermediate results, and any questions or pending items. This state is crucial for maintaining context across turns and is updated BEFORE this JSON block is generated.",
  "next_action": {
    "action_type": "CHOSEN_ACTION_TYPE_STRING",
    "action_parameters": {
      "param1_for_action": "value1",
      "param2_for_action": "value2"
    }
  }
}
"""
    manifest += "```\n"
    manifest += "IMPORTANT: After your internal reasoning and state update, output ONLY the ```json ... ``` block and then STOP. No additional text or pleasantries after the closing ``` of the JSON block.\n"
    return manifest

# --- Flask HTML Template ---
HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><title>Autonomous Multi-Model System</title>
<style>
body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Oxygen,Ubuntu,Cantarell,"Open Sans","Helvetica Neue",sans-serif;margin:20px;background:#f0f2f5;display:flex;flex-direction:column;align-items:center;color:#333}
.container{background:white;padding:25px;border-radius:10px;box-shadow:0 4px 12px rgba(0,0,0,0.1);width:95%;max-width:1200px}
h1{color:#1a73e8;text-align:center; margin-bottom: 20px;}
h2{color:#5f6368;margin-top:25px;border-bottom:1px solid #e0e0e0;padding-bottom:5px}
textarea{width:calc(100% - 22px);padding:10px;margin-bottom:12px;border:1px solid #dadce0;border-radius:6px;min-height:100px;font-size:1em;resize:vertical}
button{background:#1a73e8;color:white;padding:12px 20px;border:none;border-radius:6px;cursor:pointer;font-size:1em;transition:background-color 0.2s; min-width: 150px;}
button:hover{background:#1558b0}
button:disabled{background:#ccc;cursor:not-allowed}
.response-area,.debug-info{margin-top:20px;padding:18px;border:1px solid #e8eaed;border-radius:6px;background:#f8f9fa;white-space:pre-wrap;word-wrap:break-word;font-family:Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;font-size:0.9em}
.debug-info{max-height:450px;overflow-y:auto; list-style-type: none; padding-left: 0;}
.debug-info li { margin-bottom: 6px; padding: 6px; border-left: 4px solid #ccc; font-size: 0.85em; background-color: #fff; line-height: 1.4;}
.debug-info li:nth-child(odd) { background-color: #f7f7f7;}
.debug-info li strong { color: #1a73e8; }
.debug-info li .error { color: #d93025; font-weight: bold; }
.debug-info li .warn { color: #f9ab00; font-weight: bold; }
.status-text{text-align:center;margin:12px 0;font-style:italic;color:#5f6368}
.model-status-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:15px;margin-bottom:25px}
.model-card{padding:12px;border:1px solid #dadce0;border-radius:8px;font-size:0.85em;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,0.05);display:flex;flex-direction:column;justify-content:space-between}
.model-card strong { color: #202124; }
.model-primary-loaded{border-left:6px solid #34a853}
.model-primary-unloaded{border-left:6px solid #ea4335}
.tool-server-running{color:#34a853;font-weight:bold}
.tool-server-stopped{color:#ea4335;font-weight:500;}
.model-selector, .endpoint-selector, .collaboration-toggle {margin-bottom:15px; display: flex; align-items: center; gap: 10px;}
select{padding:8px;border:1px solid #dadce0;border-radius:4px;font-size:1em;width:100%;max-width:350px}
.collaboration-toggle label{display:flex;align-items:center;gap:8px;cursor:pointer}
.form-controls { display: flex; justify-content: space-between; align-items: center; margin-top: 15px; flex-wrap: wrap; gap: 15px;}
</style>
</head>
<body>
<div class="container">
    <h1>Autonomous Multi-Model System</h1>
    <div id="modelStatusGrid" class="model-status-grid">Loading model statuses...</div>
    
    <form id="requestForm">
        <div class="endpoint-selector">
            <label for="endpointSelect">Interaction Mode:</label>
            <select id="endpointSelect" name="endpointSelect">
                <option value="/collaborate" selected>Collaborative Agent (/collaborate)</option>
                <option value="/ask">Orchestrator Loop (/ask)</option>
            </select>
        </div>

        <div class="model-selector">
            <label for="primaryModel">Primary Model (for /collaborate):</label>
            <select id="primaryModel" name="primaryModel" required>
                <option value="">Select primary model...</option>
            </select>
        </div>
        
        <div class="collaboration-toggle">
            <label>
                <input type="checkbox" id="enableCollaboration" checked>
                Enable Peer Collaboration (for /collaborate mode)
            </label>
        </div>
        
        <label for="prompt">User Request:</label>
        <textarea id="prompt" name="prompt" placeholder="Enter your request here..." required></textarea>
        
        <div class="form-controls">
            <button type="submit" id="submitBtn" disabled>Process Request</button>
            <div id="statusText" class="status-text">Initializing...</div>
        </div>
    </form>
    
    <h2>System Response:</h2>
    <div id="responseArea" class="response-area">Waiting for response...</div>
    <h2>Execution Log / Debug Trace:</h2>
    <ul id="debugInfo" class="debug-info"></ul>
</div>

<script>
const requestForm = document.getElementById("requestForm");
const primaryModelSelect = document.getElementById("primaryModel");
const endpointSelect = document.getElementById("endpointSelect");
const enableCollaborationCheck = document.getElementById("enableCollaboration");
const modelStatusGrid = document.getElementById("modelStatusGrid");
const submitBtn = document.getElementById("submitBtn");
const statusText = document.getElementById("statusText");
const responseArea = document.getElementById("responseArea");
const debugInfoList = document.getElementById("debugInfo");

let orchestratorInternalState = null;
let orchestratorConversationLog = [];

function updatePrimaryModelSelectorVisibility() {
    const selectedEndpoint = endpointSelect.value;
    const primaryModelLabel = document.querySelector("label[for='primaryModel']");
    const collaborationToggleDiv = document.querySelector(".collaboration-toggle");

    if (selectedEndpoint === "/ask") {
        primaryModelSelect.parentElement.style.display = "none";
        if (primaryModelLabel) primaryModelLabel.style.display = "none";
        collaborationToggleDiv.style.display = "none";
        fetch("/status").then(res => res.json()).then(data => {
            const orchestrator = data.models.find(m => m.is_orchestrator);
            if (orchestrator) {
                statusText.textContent = `Ready for Orchestrator Loop (using ${orchestrator.name}).`;
            } else {
                statusText.textContent = "Orchestrator model not found. /ask mode may not function.";
            }
        });

    } else {
        primaryModelSelect.parentElement.style.display = "flex";
         if (primaryModelLabel) primaryModelLabel.style.display = "block";
        collaborationToggleDiv.style.display = "block";
        statusText.textContent = primaryModelSelect.children.length > 1 ? 
            "Select a primary model for collaboration." : 
            "No primary models loaded.";
    }
}

endpointSelect.addEventListener("change", updatePrimaryModelSelectorVisibility);


async function fetchModelStatuses() {
    try {
        const response = await fetch("/status");
        const data = await response.json();
        
        modelStatusGrid.innerHTML = "";
        const currentPrimaryModel = primaryModelSelect.value;
        primaryModelSelect.innerHTML = '<option value="">Select primary model...</option>';
        
        let primaryModelsLoaded = 0;
        
        data.models.forEach(model => {
            const card = document.createElement("div");
            card.classList.add("model-card");
            if (model.primary_generator_loaded) {
                card.classList.add("model-primary-loaded");
                const option = document.createElement("option");
                option.value = model.name;
                option.textContent = `${model.name} (${model.type}, Role: ${model.role.substring(0,30)}...)`;
                primaryModelSelect.appendChild(option);
                primaryModelsLoaded++;
            } else {
                card.classList.add("model-primary-unloaded");
            }

            const toolStatus = model.is_tool_server_candidate ? 
                (model.grpc_tool_server_running ? 
                    `<span class="tool-server-running">gRPC Tool: Port ${model.grpc_port || 'Active'}</span>` : 
                    `<span class="tool-server-stopped">gRPC Tool: Offline</span>`) : 
                "Not a gRPC tool candidate";
            
            card.innerHTML = `
                <div><strong>${model.name}</strong> <small>(${model.type}) ${model.is_orchestrator ? '(Orchestrator)' : ''}</small></div>
                <div>Flask Generator: ${model.primary_generator_loaded ? "<strong style='color:#34a853'>Loaded</strong>" : "<strong style='color:#ea4335'>Not Loaded</strong>"}</div>
                <div>Tool Server: ${toolStatus}</div>
                <small style="color:#777; margin-top:5px; display:block;">Role: ${model.role.substring(0,100)}...</small>
            `;
            modelStatusGrid.appendChild(card);
        });
        
        if (currentPrimaryModel && Array.from(primaryModelSelect.options).some(opt => opt.value === currentPrimaryModel)) {
            primaryModelSelect.value = currentPrimaryModel;
        }

        submitBtn.disabled = !(primaryModelsLoaded > 0 || endpointSelect.value === "/ask");
        updatePrimaryModelSelectorVisibility();
            
    } catch (error) {
        console.error("Error fetching statuses:", error);
        statusText.textContent = "Error fetching model statuses from server.";
    }
}

requestForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    const prompt = document.getElementById("prompt").value;
    const selectedEndpoint = endpointSelect.value;
    const primaryModel = primaryModelSelect.value;
    const enableCollaboration = enableCollaborationCheck.checked;
    
    if (!prompt) {
        alert("Please enter a user request.");
        return;
    }
    if (selectedEndpoint === "/collaborate" && !primaryModel) {
        alert("Please select a primary model for /collaborate mode.");
        return;
    }
    
    submitBtn.disabled = true;
    statusText.textContent = "Processing request via " + selectedEndpoint + "...";
    responseArea.textContent = "Working...";
    debugInfoList.innerHTML = "";
    
    const liInit = document.createElement("li");
    liInit.textContent = `Initiating request to ${selectedEndpoint}. User prompt: "${prompt.substring(0,100)}..."`;
    debugInfoList.appendChild(liInit);

    let requestBody = { prompt: prompt };
    if (selectedEndpoint === "/collaborate") {
        requestBody.primary_model = primaryModel;
        requestBody.enable_collaboration = enableCollaboration;
    } else if (selectedEndpoint === "/ask") {
        if (orchestratorInternalState) {
            requestBody.orchestrator_internal_state = orchestratorInternalState;
        }
        if (orchestratorInternalState && orchestratorConversationLog.length > 0) {
            orchestratorConversationLog.push(`USER_FOLLOW_UP_REQUEST: ${prompt}`);
            requestBody.conversation_log = orchestratorConversationLog;
        } else {
            orchestratorInternalState = null; 
            orchestratorConversationLog = [`USER_REQUEST: ${prompt}`];
            requestBody.conversation_log = orchestratorConversationLog;
        }
    }
    
    try {
        const response = await fetch(selectedEndpoint, { 
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        
        (data.debug_trace || []).forEach(trace_entry => {
            const li = document.createElement("li");
            let display_trace = String(trace_entry).replace(/\\n/g, "<br>");
            if (display_trace.includes("ERROR:") || display_trace.includes("CRITICAL")) {
                 li.innerHTML = `<span class="error">${display_trace}</span>`;
            } else if (display_trace.includes("WARN:") || display_trace.includes("Warning:")) {
                 li.innerHTML = `<span class="warn">${display_trace}</span>`;
            } else if (display_trace.startsWith("---") || display_trace.startsWith("===")) {
                 li.innerHTML = `<strong>${display_trace}</strong>`;
            } else {
                 li.innerHTML = display_trace;
            }
            debugInfoList.appendChild(li);
        });
        
        if (response.ok) {
            responseArea.textContent = data.final_answer || data.answer || "No specific answer field in response.";
            statusText.textContent = `Request to ${selectedEndpoint} complete.`;
            if (selectedEndpoint === "/ask") {
                orchestratorInternalState = data.orchestrator_internal_state;
                orchestratorConversationLog = data.conversation_log || [];
                if (data.final_answer && (data.final_answer.includes("Max iterations reached") || data.final_answer.includes("FINAL_RESPONSE action received"))) {
                    const liDone = document.createElement("li");
                    liDone.innerHTML = "<strong>Orchestrator loop concluded. Submit new request or refresh for a new session.</strong>";
                    debugInfoList.appendChild(liDone);
                } else {
                     const liNext = document.createElement("li");
                    liNext.innerHTML = "<strong>Orchestrator state updated. You can provide a follow-up prompt.</strong>";
                    debugInfoList.appendChild(liNext);
                }
            }
        } else {
            responseArea.textContent = `Error: ${data.error || "Unknown error from server."}`;
            statusText.textContent = `Error processing request at ${selectedEndpoint}.`;
             if (selectedEndpoint === "/ask") {
                orchestratorInternalState = null; 
                orchestratorConversationLog = [];
            }
        }
        
    } catch (error) {
        console.error("Client-side error:", error);
        statusText.textContent = "Network or client-side script error.";
        responseArea.textContent = `Client Error: ${error.message}`;
        const liErr = document.createElement("li");
        liErr.innerHTML = `<span class="error">Client Error: ${error.message}</span>`;
        debugInfoList.appendChild(liErr);
         if (selectedEndpoint === "/ask") {
            orchestratorInternalState = null; 
            orchestratorConversationLog = [];
        }
    } finally {
        submitBtn.disabled = false;
        fetchModelStatuses();
    }
});

window.addEventListener("load", () => {
    fetchModelStatuses();
    setInterval(fetchModelStatuses, 25000);
});
</script>
</body></html>"""

# --- Flask Routes ---
@app.route('/')
def home():
    """Renders the main HTML user interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/status', methods=['GET'])
def get_status_route():
    """Returns the current status of all models (loaded, tool server running, etc.)."""
    models_status = []
    for m_def in MODEL_INVENTORY:
        is_tool_server_process_running = False
        if m_def.is_tool_server_candidate and m_def.grpc_server_process:
            is_tool_server_process_running = m_def.grpc_server_process.poll() is None

        models_status.append({
            "name": m_def.name,
            "path": os.path.basename(m_def.actual_path) if m_def.actual_path else m_def.model_id_or_path,
            "type": m_def.type_config,
            "n_ctx": m_def.n_ctx,
            "primary_generator_loaded": m_def.instance is not None,
            "is_tool_server_candidate": m_def.is_tool_server_candidate,
            "grpc_tool_server_running": is_tool_server_process_running,
            "grpc_port": m_def.grpc_port if is_tool_server_process_running else None,
            "role": m_def.role_description,
            "is_orchestrator": m_def.is_orchestrator,
            "loading_strategy": m_def.loading_strategy
        })
    return jsonify({"models": models_status})

@app.route('/list_main_agents', methods=['GET'])
def list_main_agents():
    """Returns agent details for the aggregator in the expected format."""
    agents_details = []
    for model_def in MODEL_INVENTORY:
        agents_details.append({
            "name": model_def.name,
            "type": "orchestrator_main_server" if model_def.is_orchestrator else "specialist_main_server",
            "type_config": model_def.type_config,
            "role_description": model_def.role_description,
            "n_ctx": model_def.n_ctx,
            "loading_strategy": model_def.loading_strategy,
            "is_orchestrator": model_def.is_orchestrator,
            "loaded": model_def.instance is not None,
            "path_config": model_def.model_id_or_path,
            "actual_path": model_def.actual_path,
            "port": None  # This server is Flask-based, not port-specific per model
        })
    return jsonify(agents_details)

def extract_json_from_text(text_blob: str) -> dict | None:
    """Extracts a JSON object from a text blob, specifically looking for ```json ... ``` blocks."""
    text_blob = text_blob.strip()
    match_block = re.search(r"```json\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text_blob, re.DOTALL)
    if match_block:
        json_str_candidate = match_block.group(1).strip()
        try:
            return json.loads(json_str_candidate)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError within ```json ... ``` block: {e}. Candidate (first 200 chars): '{json_str_candidate[:200]}...'")
    
    first_brace = text_blob.find('{')
    last_brace = text_blob.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str_candidate = text_blob[first_brace : last_brace + 1]
        try:
            if json_str_candidate.strip().startswith("{") and ':' in json_str_candidate:
                return json.loads(json_str_candidate)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError with general brace matching fallback: {e}. Candidate (first 200 chars): '{json_str_candidate[:200]}...'")
            
    return None


@app.route('/collaborate', methods=['POST'])
def collaborate_route():
    """
    Handles collaborative multi-agent requests. A primary model starts a collaboration chain,
    potentially delegating to other models (peers) via [CONSULT_PEER] tags.
    """
    data = request.get_json()
    user_input = data.get("prompt", "").strip()
    primary_model_name = data.get("primary_model", "")
    enable_collaboration = data.get("enable_collaboration", True)
    
    if not user_input: return jsonify({"error": "No input prompt provided"}), 400
    if not primary_model_name: return jsonify({"error": "No primary_model specified for collaboration"}), 400
    
    debug_log_collab = [f"COLLABORATE_ENDPOINT_INIT: User Input '{user_input[:70]}...', Primary Model: {primary_model_name}, Collaboration Enabled: {enable_collaboration}"]
    start_time = time.time()
    
    primary_def = next((m for m in MODEL_INVENTORY if m.name == primary_model_name), None)
    if not primary_def:
        err_msg = f"Primary model '{primary_model_name}' not found in inventory."
        debug_log_collab.append(f"ERROR: {err_msg}")
        return jsonify({"error": err_msg, "debug_trace": debug_log_collab}), 404
    
    try:
        get_model_instance_by_name(primary_def.name)
        if not primary_def.instance:
             raise RuntimeError(f"Flask-hosted instance for primary model {primary_def.name} could not be ensured.")

        debug_log_collab.append(f"Primary model {primary_model_name} Flask-hosted instance confirmed.")
        
        response_text = generate_with_llm_instance(
            initial_prompt_text=user_input, 
            model_def=primary_def,
            enable_peer_collaboration=enable_collaboration,
            debug_log=debug_log_collab
        )
        
        debug_log_collab.append(f"COLLABORATE_ENDPOINT_TIME: {time.time() - start_time:.2f}s")
        
        return jsonify({
            "answer": response_text,
            "primary_model": primary_model_name,
            "collaboration_enabled": enable_collaboration,
            "debug_trace": debug_log_collab
        })
        
    except Exception as e:
        error_message = f"Error during /collaborate processing with {primary_model_name}: {str(e)}"
        import traceback
        traceback.print_exc()
        debug_log_collab.append(f"COLLABORATE_ENDPOINT_EXCEPTION: {error_message}\n{traceback.format_exc()}")
        return jsonify({
            "error": error_message,
            "debug_trace": debug_log_collab
        }), 500


@app.route('/ask', methods=['POST'])
def handle_orchestrator_loop_request():
    """
    Handles requests for the main Orchestrator loop. The Orchestrator uses tools
    (CALL_SPECIALIST_MODEL, CONTEXT_DB, SELF_GENERATE, FINAL_RESPONSE) to accomplish tasks.
    It maintains internal state and a conversation log across iterations.
    """
    debug_trace_ask = [] 
    start_time = time.time()
    
    orchestrator_def = next((m for m in MODEL_INVENTORY if m.is_orchestrator), None)
    if not orchestrator_def:
        return jsonify({"error": "Orchestrator model (is_orchestrator=True) not defined/found in inventory.", "debug_trace": ["No orchestrator model configured."]}), 503
    
    debug_trace_ask.append(f"ASK_ENDPOINT_INIT: Using Orchestrator '{orchestrator_def.name}'.")
    try:
        get_model_instance_by_name(orchestrator_def.name)
        if not orchestrator_def.instance:
            raise RuntimeError(f"Flask-hosted instance for orchestrator {orchestrator_def.name} could not be ensured.")
    except Exception as e:
        err_msg = f"Orchestrator model '{orchestrator_def.name}' Flask-hosted instance loading error: {e}"
        debug_trace_ask.append(f"CRITICAL_ERROR: {err_msg}")
        return jsonify({"error": err_msg, "debug_trace": debug_trace_ask}), 503

    data = request.get_json()
    user_input_text = data.get("prompt", "").strip()
    orchestrator_internal_state = data.get("orchestrator_internal_state", f"New Task. Original User Request: {user_input_text}")
    current_session_interaction_log = data.get("conversation_log", [])

    if not current_session_interaction_log:
        current_session_interaction_log.append(f"USER_REQUEST_INITIATED: {user_input_text}")
    elif user_input_text and (not current_session_interaction_log or user_input_text not in current_session_interaction_log[-1]):
        current_session_interaction_log.append(f"USER_INPUT_FOLLOW_UP: {user_input_text}")

    if not any(log_entry.startswith("USER_REQUEST") for log_entry in current_session_interaction_log) and not user_input_text :
        err_msg = "Input prompt/question missing and no initial user request in conversation history for Orchestrator loop."
        debug_trace_ask.append(f"ERROR: {err_msg}")
        return jsonify({"error": err_msg, "debug_trace": debug_trace_ask}), 400

    debug_trace_ask.append(f"Orchestrator Loop Start. Initial User Input for this turn: '{user_input_text[:100]}...'. Initial State len: {len(orchestrator_internal_state)}. Log items: {len(current_session_interaction_log)}")
    
    iteration_count = 0
    consecutive_json_errors = 0
    final_answer_from_orchestrator = None
    tool_manifest = build_orchestrator_tool_manifest()

    while iteration_count < MAX_ORCHESTRATOR_ITERATIONS:
        iteration_count += 1
        debug_trace_ask.append(f"\n--- ORCHESTRATOR LOOP ITERATION {iteration_count} / {MAX_ORCHESTRATOR_ITERATIONS} ---")
        
        recent_history_str = "\n".join(current_session_interaction_log[-6:])
        
        orchestrator_prompt_for_llm = (
            f"{tool_manifest}\n\n"
            f"CURRENT COMPREHENSIVE INTERNAL STATE (Your memory, previous plans, notes, context DB IDs):\n---\n{orchestrator_internal_state}\n---\n\n"
            f"RECENT INTERACTIONS IN THIS ORCHESTRATION SESSION (User inputs and Tool outputs):\n---\n{recent_history_str}\n---\n\n"
            f"TASK FOR THIS TURN:\nBased on the original user request (see log), your comprehensive internal state, and the recent interactions, "
            f"first, write down your 'orchestrator_reasoning' (your thought process for this step). "
            f"Second, update your 'orchestrator_internal_state' with any new information, completed steps, or changes to your overall plan. "
            f"Third, formulate the 'next_action' as a JSON object to call a tool or provide the final answer. "
            f"Combine these into a single JSON response block prefixed by ```json and ending with ```. This JSON block is your ONLY output."
        )

        directives_json = None
        raw_orchestrator_llm_output = ""
        try:
            raw_orchestrator_llm_output = generate_with_llm_instance(
                orchestrator_prompt_for_llm,
                orchestrator_def, 
                temperature_override=0.4,
                enable_peer_collaboration=False,
                debug_log=debug_trace_ask
            )
            debug_trace_ask.append(f"ORCH_RAW_LLM_OUTPUT (Iter {iteration_count}):\n{raw_orchestrator_llm_output}")
            
            directives_json = extract_json_from_text(raw_orchestrator_llm_output)
            if directives_json is None:
                error_msg_for_llm_feedback = f"[SYSTEM_FEEDBACK_TO_ORCHESTRATOR: Your response did not contain a single, valid JSON object enclosed in ```json ... ``` with required keys. This error is critical for task progression. You must correct this in the next turn. Raw output for reference: {raw_orchestrator_llm_output[:500]} (truncated)]"
                debug_trace_ask.append(f"ERROR: Orchestrator (Iter {iteration_count}) failed to produce valid JSON. Error: {error_msg_for_llm_feedback}")
                current_session_interaction_log.append(f"SYSTEM_ERROR_ORCHESTRATOR_OUTPUT (Iter {iteration_count-1}): JSON parsing failed. {error_msg_for_llm_feedback}")
                orchestrator_internal_state += f"\n[Self-Correction Note (Iter {iteration_count}): My previous response failed JSON parsing. I MUST output a valid JSON block exactly as specified in the tools manifest. Problematic fragment from my last output: {raw_orchestrator_llm_output[:200]}]"
                consecutive_json_errors += 1
                if consecutive_json_errors >= MAX_JSON_PARSE_FAILURES:
                    final_answer_from_orchestrator = f"Orchestrator failed to produce valid JSON output after {consecutive_json_errors} consecutive attempts. Please refine your instruction or restart the task. Last raw output: {raw_orchestrator_llm_output}"
                    break
                time.sleep(0.2)
                continue

            required_top_keys = ["orchestrator_reasoning", "orchestrator_internal_state", "next_action"]
            if not all(key in directives_json for key in required_top_keys):
                missing_keys = [key for key in required_top_keys if key not in directives_json]
                raise ValueError(f"Orchestrator JSON missing required top-level keys: {', '.join(missing_keys)}. Found: {list(directives_json.keys())}")
            
            next_action_val = directives_json.get("next_action")
            if not isinstance(next_action_val, dict) or \
               "action_type" not in next_action_val or \
               not isinstance(next_action_val.get("action_parameters"), dict):
                raise ValueError("Orchestrator JSON 'next_action' is malformed (must be a dictionary with 'action_type' (string) and 'action_parameters' (dictionary)).")

            consecutive_json_errors = 0

        except (json.JSONDecodeError, ValueError) as e_json:
            error_context = f"Orchestrator output structure issue (Iter {iteration_count}): {e_json}. Raw output snippet: {raw_orchestrator_llm_output[:300]}..."
            debug_trace_ask.append(f"ERROR: {error_context}")
            current_session_interaction_log.append(f"SYSTEM_ERROR_ORCHESTRATOR_OUTPUT (Iter {iteration_count-1}): {error_context}")
            orchestrator_internal_state += f"\n[System Self-Correction Note (Iter {iteration_count}): My previous JSON output was problematic: {e_json}. I must provide all required fields and valid JSON. Problematic text: {raw_orchestrator_llm_output[:200]}]"
            consecutive_json_errors += 1
            if consecutive_json_errors >= MAX_JSON_PARSE_FAILURES:
                final_answer_from_orchestrator = f"Orchestrator produced malformed JSON output repeatedly after {consecutive_json_errors} attempts. Last error: {e_json}"
                break
            time.sleep(0.2)
            continue
        except Exception as e_gen:
            err_msg = f"Orchestrator LLM generation call failed unexpectedly (Iter {iteration_count}): {e_gen}"
            debug_trace_ask.append(f"CRITICAL_ERROR: {err_msg}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": err_msg, "final_answer": None, "debug_trace": debug_trace_ask, "orchestrator_internal_state": orchestrator_internal_state, "conversation_log": current_session_interaction_log}), 500

        orchestrator_internal_state = directives_json.get("orchestrator_internal_state", orchestrator_internal_state)
        orchestrator_reasoning_text = directives_json.get("orchestrator_reasoning", "No reasoning provided by orchestrator.")
        current_session_interaction_log.append(f"ORCHESTRATOR_REASONING (Iter {iteration_count}):\n{orchestrator_reasoning_text}")
        debug_trace_ask.append(f"ORCH_REASONING (Parsed - Iter {iteration_count}): {orchestrator_reasoning_text[:200]}...")

        next_action_details = directives_json["next_action"]
        action_type_chosen = next_action_details.get("action_type")
        action_params_chosen = next_action_details.get("action_parameters", {})
        
        tool_call_info_for_log = f"ORCHESTRATOR_ACTION (Iter {iteration_count}): Type='{action_type_chosen}', Params='{json.dumps(action_params_chosen)}'"
        debug_trace_ask.append(tool_call_info_for_log)
        current_session_interaction_log.append(tool_call_info_for_log)
        
        tool_output_for_next_turn = f"SYSTEM_NOTE: Action '{action_type_chosen}' acknowledged. Preparing to execute..."

        try:
            if action_type_chosen == "CALL_SPECIALIST_MODEL":
                specialist_name = action_params_chosen.get("specialist_model_name")
                task_for_specialist = action_params_chosen.get("task_for_specialist")
                if not specialist_name or not task_for_specialist:
                    raise ValueError("Missing 'specialist_model_name' or 'task_for_specialist' for CALL_SPECIALIST_MODEL action.")
                
                specialist_def = next((m for m in MODEL_INVENTORY if m.name == specialist_name and not m.is_orchestrator), None)
                if not specialist_def:
                    raise ValueError(f"Specialist model '{specialist_name}' not found in inventory or is the orchestrator itself.")
                
                get_model_instance_by_name(specialist_name)
                if not specialist_def.instance:
                    raise RuntimeError(f"Failed to get/load Flask-hosted instance for specialist model '{specialist_name}'.")
                
                specialist_specific_debug_log = [f"SPECIALIST_CALL_INIT ({specialist_name}): Task '{task_for_specialist[:70]}...'"]
                specialist_response_text = generate_with_llm_instance(
                    initial_prompt_text=task_for_specialist, 
                    model_def=specialist_def, 
                    enable_peer_collaboration=True,
                    debug_log=specialist_specific_debug_log
                )
                debug_trace_ask.extend([f"  ↪ {log_item}" for log_item in specialist_specific_debug_log])
                tool_output_for_next_turn = f"TOOL_OUTPUT (RESPONSE_FROM_SPECIALIST_{specialist_name.upper()}):\n{specialist_response_text}"

            elif action_type_chosen == "STORE_IN_CONTEXT_DB":
                storage_id = action_params_chosen.get("unique_storage_id")
                content_to_store_val = action_params_chosen.get("content_to_store")
                if not storage_id or content_to_store_val is None:
                    raise ValueError("Missing 'unique_storage_id' or 'content_to_store' for STORE_IN_CONTEXT_DB action.")
                success_flag, status_message = store_context_in_db(storage_id, content_to_store_val)
                tool_output_for_next_turn = f"TOOL_OUTPUT (CONTEXT_DB_STORE_RESULT, ID: '{storage_id}'): {status_message}"

            elif action_type_chosen == "RETRIEVE_FROM_CONTEXT_DB":
                id_to_query = action_params_chosen.get("unique_storage_id_or_query")
                if not id_to_query:
                    raise ValueError("Missing 'unique_storage_id_or_query' for RETRIEVE_FROM_CONTEXT_DB action.")
                retrieved_data, status_message = retrieve_context_from_db(id_to_query)
                retrieved_data_str = retrieved_data if retrieved_data is not None else "Content not found or error during retrieval."
                tool_output_for_next_turn = f"TOOL_OUTPUT (CONTEXT_DB_RETRIEVE_RESULT, ID: '{id_to_query}'): {status_message}\nCONTENT_RETRIEVED:\n{retrieved_data_str}"

            elif action_type_chosen == "REQUEST_SUMMARY":
                text_to_be_summarized = action_params_chosen.get("text_to_summarize")
                if not text_to_be_summarized:
                    raise ValueError("Missing 'text_to_summarize' for REQUEST_SUMMARY action.")
                summarizer_model_override = action_params_chosen.get("target_summarizer_model")
                custom_summary_instructions = action_params_chosen.get("summarization_instructions")
                summary_text_result, status_message = summarize_text_with_model(text_to_be_summarized, summarizer_model_override, custom_summary_instructions)
                tool_output_for_next_turn = f"TOOL_OUTPUT (SUMMARY_RESULT, Status: {status_message}):\n{summary_text_result}"

            elif action_type_chosen == "SELF_GENERATE":
                generated_text_segment = action_params_chosen.get('generated_segment', 'Orchestrator generated no specific text for SELF_GENERATE.')
                tool_output_for_next_turn = f"TOOL_OUTPUT (ORCHESTRATOR_SELF_GENERATED_SEGMENT):\n{generated_text_segment}"

            elif action_type_chosen == "FINAL_RESPONSE":
                final_answer_content_val = action_params_chosen.get("final_answer_content")
                if final_answer_content_val is None:
                    raise ValueError("Missing 'final_answer_content' for FINAL_RESPONSE action.")
                final_answer_from_orchestrator = final_answer_content_val
                debug_trace_ask.append(f"--> FINAL_ANSWER received from Orchestrator action: '{str(final_answer_from_orchestrator)[:150]}...'")
                tool_output_for_next_turn = f"SYSTEM_NOTE: FINAL_RESPONSE action received. Orchestration loop will terminate after this iteration."
                current_session_interaction_log.append(tool_output_for_next_turn)
                break
            
            else:
                raise ValueError(f"Unknown action_type: '{action_type_chosen}'. Please use one of the documented action types.")
            
            debug_trace_ask.append(f"TOOL_EXEC_RESULT (Action: {action_type_chosen}, Iter {iteration_count}): {str(tool_output_for_next_turn)[:250]}...")

        except Exception as e_action:
            error_message_for_log_and_llm = f"ERROR_EXECUTING_ACTION (Type: {action_type_chosen}, Params: {json.dumps(action_params_chosen)}): {str(e_action)}. Orchestrator, you need to analyze this error and decide the next action (e.g., retry with different parameters, try a different approach, or ask user for clarification)."
            debug_trace_ask.append(error_message_for_log_and_llm)
            tool_output_for_next_turn = error_message_for_log_and_llm

        current_session_interaction_log.append(tool_output_for_next_turn)

    if final_answer_from_orchestrator is None:
        final_answer_from_orchestrator = (f"Max iterations ({MAX_ORCHESTRATOR_ITERATIONS}) reached. "
                                          f"Orchestration did not produce a FINAL_RESPONSE action. "
                                          f"Last known internal state: {orchestrator_internal_state}")
        debug_trace_ask.append(f"WARNING: Max iterations ({MAX_ORCHESTRATOR_ITERATIONS}) reached without a FINAL_RESPONSE action from orchestrator.")

    final_answer_str = json.dumps(final_answer_from_orchestrator, indent=2) if isinstance(final_answer_from_orchestrator, (dict, list)) else str(final_answer_from_orchestrator)

    total_time = time.time() - start_time
    debug_trace_ask.append(f"\n--- Orchestrator Loop Total Execution Time: {total_time:.2f} seconds ---")
    
    return jsonify({
        "final_answer": final_answer_str,
        "debug_trace": debug_trace_ask,
        "orchestrator_internal_state": orchestrator_internal_state,
        "conversation_log": current_session_interaction_log
    })


def _direct_model_query(model_name: str, request_data: dict, endpoint_name_for_log: str):
    """Handles direct queries to a model, potentially with collaboration if enabled and appropriate."""
    input_text = request_data.get("prompt", "").strip()
    if not input_text:
        return jsonify({"error": f"Direct query to {model_name} ({endpoint_name_for_log}): input prompt/question missing."}), 400

    model_def = next((m for m in MODEL_INVENTORY if m.name == model_name), None)
    if not model_def:
        return jsonify({"error": f"Direct query ({endpoint_name_for_log}): Model '{model_name}' not found."}), 404
    
    debug_log_direct = [f"DIRECT_QUERY_INIT to {model_name} for endpoint {endpoint_name_for_log}. User Input: '{input_text[:70]}'"]
    try:
        get_model_instance_by_name(model_def.name)
        if not model_def.instance:
            return jsonify({"error": f"Direct query ({endpoint_name_for_log}): Model '{model_def.name}' Flask-hosted generator instance failed to load."}), 503

        temperature = request_data.get("temperature", FIXED_TEMPERATURE)
        
        enable_collaboration_param = request_data.get("enable_collaboration", True)
        
        effective_enable_collaboration = enable_collaboration_param and not model_def.is_orchestrator
        
        if model_def.is_orchestrator and enable_collaboration_param:
             debug_log_direct.append("Note: Orchestrator model queried directly via its own endpoint. Collaboration chain disabled for this mode; providing direct response. Use /ask or /collaborate for full orchestration/collaboration.")


        is_simple_request_heuristic = (
            len(input_text.split()) < 25 or
            any(word in input_text.lower() for word in ["quickly", "simple", "brief", "short", "outline", "summarize", "what is", "define"])
        )

        if effective_enable_collaboration and not is_simple_request_heuristic:
            debug_log_direct.append(f"Using full collaborative system (generate_with_llm_instance) for model {model_name}.")
            resp_txt = generate_with_llm_instance(
                initial_prompt_text=input_text, 
                model_def=model_def, 
                temperature_override=temperature,
                enable_peer_collaboration=True,
                debug_log=debug_log_direct
            )
        else:
            if not effective_enable_collaboration:
                debug_log_direct.append(f"Using simple_generate for model {model_name} as collaboration is disabled or model is orchestrator.")
            elif is_simple_request_heuristic:
                 debug_log_direct.append(f"Using simple_generate for model {model_name} due to simple request heuristic.")
            resp_txt = simple_generate(input_text, model_def, max_tokens=500, temperature=temperature)

        return jsonify({"answer": resp_txt, f"debug_trace": debug_log_direct})
    except Exception as e:
        error_msg = f"Direct query to {model_name} ({endpoint_name_for_log}) encountered an error: {str(e)}"
        import traceback
        traceback.print_exc()
        debug_log_direct.append(f"DIRECT_QUERY_EXCEPTION: {error_msg}\n{traceback.format_exc()}")
        return jsonify({"error": error_msg, f"debug_trace": debug_log_direct}), 500


def create_direct_agent_endpoints():
    """Dynamically creates POST endpoints for each model by its lowercased name."""
    print("\n--- Creating Direct Agent Query Endpoints ---")
    for model_def_item in MODEL_INVENTORY:
        endpoint_path = f"/{model_def_item.name.lower().replace(' ', '_').replace('-', '_')}"
        endpoint_function_name = f"direct_query_for_{model_def_item.name.lower().replace(' ', '_').replace('-', '_')}"

        def create_route_function(name_for_closure):
            def dynamic_route_function():
                return _direct_model_query(name_for_closure, request.get_json(), name_for_closure)
            return dynamic_route_function

        route_func_object = create_route_function(model_def_item.name)
        route_func_object.__name__ = endpoint_function_name

        app.add_url_rule(
            rule=endpoint_path,
            endpoint=endpoint_function_name,
            view_func=route_func_object, 
            methods=['POST']
        )
        print(f"Created direct query endpoint: POST {endpoint_path} for model '{model_def_item.name}' (handler: {endpoint_function_name})")

# --- Server Initialization ---
if __name__ == '__main__':
    print("--- Starting Autonomous Multi-Model System ---")
    if not PYYAML_AVAILABLE:
        sys.exit("CRITICAL FAILURE: PyYAML is required for llama-server YAML configuration but not found. Please install it: pip install PyYAML.")

    if not GRPC_AVAILABLE:
        print("CRITICAL WARNING: gRPC components were not available or stubs not generated. gRPC tool functionality will fail.")
        # If the script did not exit previously, this warning indicates gRPC won't work.

    ensure_context_db_dir()
    resolve_and_load_models()

    print("\n--- Initializing gRPC Tool Servers (for GGUF models marked as tool_server_candidate) ---")
    successful_tool_servers_count = 0
    tool_server_candidates = [md for md in MODEL_INVENTORY if md.is_tool_server_candidate and md.type_config == 'gguf']
    
    if tool_server_candidates:
        print(f"Found {len(tool_server_candidates)} GGUF model(s) marked as potential gRPC tool servers.")
        for model_to_start_as_tool in tool_server_candidates:
            print(f"Attempting to initialize gRPC tool server for: {model_to_start_as_tool.name}")
            if model_to_start_as_tool.actual_path and os.path.exists(model_to_start_as_tool.actual_path):
                if start_llama_cpp_tool_server(model_to_start_as_tool):
                    successful_tool_servers_count += 1
            else:
                print(f"Skipping tool server for {model_to_start_as_tool.name}: model file not found at '{model_to_start_as_tool.actual_path}'.")
        
        if successful_tool_servers_count > 0:
            print(f"Successfully launched or confirmed {successful_tool_servers_count} gRPC tool server(s).")
        else:
            print(f"Warning: Failed to launch any of the {len(tool_server_candidates)} gRPC tool server candidate(s). gRPC calls to these tools will likely fail.")
    else:
        print("No GGUF models are configured as gRPC tool server candidates in MODEL_INVENTORY.")


    create_direct_agent_endpoints()

    orchestrator_model_for_ask = next((m for m in MODEL_INVENTORY if m.is_orchestrator), None)
    if not orchestrator_model_for_ask:
        print(f"\nCRITICAL WARNING: No model in MODEL_INVENTORY has 'is_orchestrator=True'. The /ask endpoint (Orchestrator loop) will NOT function.")
    elif not orchestrator_model_for_ask.instance :
         print(f"\nCRITICAL WARNING: Orchestrator model ('{orchestrator_model_for_ask.name}') Flask-hosted generator failed to load. The /ask endpoint will NOT function correctly.")
    else:
        print(f"\nOrchestrator model '{orchestrator_model_for_ask.name}' Flask-hosted generator is loaded and configured for the /ask endpoint.")

    print(f"\nFlask server starting on http://0.0.0.0:{SERVER_PORT}")
    print("Access the UI at this address in your browser.")
    print(f"--- Ensure models are placed in '{MODELS_BASE_DIRECTORY}' or its subdirectories, OR defined with absolute paths. ---")
    print(f"--- Ensure '{LLAMA_CPP_SERVER_EXECUTABLE}' is correctly pathed and executable for gRPC tools. ---")
    print("--- CAUTION: SIGNIFICANT SYSTEM RESOURCES (CPU, RAM, VRAM) WILL BE USED. ---")
    
    app.run(host='0.0.0.0', port=SERVER_PORT, debug=False, threaded=True)
