
import requests
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import uvicorn
import socket
import sqlite3
import time
import os
import json
import subprocess
import platform
import threading
import re
import signal
import shutil 
from fastapi.responses import FileResponse

# --- Configuration Constants ---
DB_NAME = "conversations_v1_5.db"
BASE_WORKSPACE_DIR = "workspaces_v1_5"
MAIN_AGENT_SERVER_URL = "http://localhost:5035" # URL for local_model.py
IS_WINDOWS = platform.system().lower().startswith("win")
AGENT_SCAN_PORTS_START = 5001
AGENT_SCAN_PORTS_END = 5040
MAIN_AGENT_SERVER_PORT = int(MAIN_AGENT_SERVER_URL.split(":")[-1]) if ":" in MAIN_AGENT_SERVER_URL else None

# --- GLOBAL CACHE FOR LOCAL_MODEL.PY AGENT DEFINITIONS ---
# This will store the raw dicts received from local_model.py's /list_main_agents.
# This cache is necessary for the aggregator to intelligently look up model details
# for agents hosted by local_model.py without re-querying it constantly.
_discovered_local_models: List[Dict[str, Any]] = []

# --- Database Initialization and Utilities ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    # Messages Table (Enhanced)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        agent_identifier TEXT, -- Can be port, name, or other ID
        content TEXT NOT NULL,
        raw_response TEXT,    -- Store raw agent response for debugging
        timestamp REAL NOT NULL
    )
    """)

    # Projects Table (Enhanced structure from V2)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        conversation_id TEXT PRIMARY KEY,
        project_json TEXT NOT NULL,
        updated_at REAL NOT NULL
    )
    """)

    # Memories Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        timestamp REAL NOT NULL
    )
    """)

    # Vectors Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS vectors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        embedding BLOB NOT NULL, -- Storing as JSON string, could be BLOB for raw bytes
        metadata TEXT NOT NULL,  -- Storing as JSON string
        timestamp REAL NOT NULL
    )
    """)

    # Pull Requests Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pull_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        pr_json TEXT NOT NULL,      -- JSON string of PR data (code blocks, description)
        status TEXT NOT NULL,     -- e.g., pending, approved, merged, closed
        timestamp REAL NOT NULL
    )
    """)

    # Structured Data Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS structured_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        data_type TEXT NOT NULL,
        json_data TEXT NOT NULL,   -- Storing complex JSON as text
        timestamp REAL NOT NULL
    )
    """)

    # Pipeline Runs Table (Enhanced from V2)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        step_name TEXT NOT NULL,
        status TEXT NOT NULL,      -- e.g., passed, failed, running
        detail TEXT,               -- Output or error details
        timestamp REAL NOT NULL
    )
    """)

    # Executed Commands Table (Enhanced from V2)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS executed_commands (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        command TEXT NOT NULL,
        start_time REAL NOT NULL,
        end_time REAL,
        stdout TEXT,
        stderr TEXT,
        return_code INTEGER,
        timestamp REAL NOT NULL  -- Timestamp of when the record was created
    )
    """)

    # Executed Code Files Table (for auto-code-run)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS executed_code_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        last_run_time REAL,
        last_return_code INTEGER,
        run_count INTEGER NOT NULL DEFAULT 0
    )
    """)

    conn.commit()
    conn.close()

# Call init_db at module level to ensure tables exist when app starts
init_db()

def get_conversation_folder(conversation_id: str) -> str:
    safe_id = re.sub(r'[^\w\-.]', '_', conversation_id)
    folder_path = os.path.join(BASE_WORKSPACE_DIR, safe_id)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def log_message_db(conversation_id: str, role: str, content: str,
                   agent_identifier: Optional[str] = None,
                   raw_response: Optional[str] = None):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO messages (conversation_id, role, agent_identifier, content, raw_response, timestamp)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (conversation_id, role, agent_identifier, content, raw_response, time.time()))
    conn.commit()
    conn.close()

# --- Pydantic Models ---
class AgentIdentifier(BaseModel):
    name: Optional[str] = None
    port: Optional[int] = None
    type: str # e.g., "specialist_main_server", "network_scanned", "orchestrator_main_server"
    role_description: Optional[str] = None # Added for frontend display
    n_ctx: Optional[int] = None # Added
    loading_strategy: Optional[str] = None # Added

class MessageRequest(BaseModel):
    agent_port: Optional[int] = None # Kept for backward compatibility if FE sends it
    agent_name: Optional[str] = None
    agent_identifier_obj: Optional[AgentIdentifier] = Field(None, alias="agentIdentifier")
    question: str
    max_tokens: Optional[int] = 10000
    conversation_id: Optional[str] = None
    temperature: Optional[float] = None

class WorkspaceData(BaseModel):
    type: str # "text", "code", "html", "css"
    title: str # filename
    content: str
    lang: Optional[str] = None # For code workspaces

class InteractiveStepRequest(BaseModel):
    conversation_id: str
    agent_port: Optional[int] = None
    agent_name: Optional[str] = None
    agent_identifier_obj: Optional[AgentIdentifier] = Field(None, alias="agentIdentifier")
    role: str
    role_description: Optional[str] = ""
    workspaces: List[WorkspaceData]
    user_message: Optional[str] = None
    last_agent_message: Optional[str] = None
    max_tokens: Optional[int] = 10000
    finish_step: bool = False
    temperature: Optional[float] = None

class ChainStep(BaseModel):
    agent_port: Optional[int] = None
    agent_name: Optional[str] = None
    agent_identifier_obj: Optional[AgentIdentifier] = Field(None, alias="agentIdentifier")
    max_tokens: Optional[int] = 10000
    role: Optional[str] = "Developer"
    role_description: Optional[str] = ""
    temperature: Optional[float] = None

class ChainRequest(BaseModel):
    chain: List[ChainStep]
    initial_prompt: str
    conversation_id: Optional[str] = None

class LogRequest(BaseModel):
    conversation_id: str
    role: str
    content: str
    agent_port: Optional[int] = None
    agent_name: Optional[str] = None
    agent_identifier_obj: Optional[AgentIdentifier] = Field(None, alias="agentIdentifier")

class ProjectData(BaseModel):
    initialPrompt: Optional[str] = ""
    steps: Optional[List[Dict[str, Any]]] = []
    workspaces: Optional[List[Dict[str, Any]]] = []
    customOrgs: Optional[List[Dict[str, Any]]] = []
    contextAgentPort: Optional[int] = None # V2 specific, useful for UI context agent selection
    env: Optional[Dict[str, str]] = None # For project-specific env variables
    pipelineSteps: Optional[List[Dict[str, str]]] = None # e.g., [{"name": "build", "cmd": "npm run build"}]

class MemoryRequest(BaseModel):
    conversation_id: str
    key: str
    value: str

class CommandExecRequest(BaseModel):
    command: str
    conversation_id: Optional[str] = None # Optional: if command should run in a convo context

class PullRequestData(BaseModel):
    conversation_id: str
    code_blocks: List[Dict[str, str]] # Expects list of {"title": ..., "content": ...}
    description: Optional[str] = ""

class StructuredData(BaseModel):
    conversation_id: str
    data_type: str
    json_data: Dict[str, Any]

class VectorData(BaseModel):
    conversation_id: str
    embedding: List[float]
    metadata: Dict[str, Any]

class PipelineRequest(BaseModel):
    conversation_id: str
    build_command: Optional[str] = "echo 'No build command specified in request.'"
    test_command: Optional[str] = "echo 'No test command specified in request.'"
    lint_command: Optional[str] = "echo 'No lint command specified in request.'"

# --- FastAPI App Setup ---
app = FastAPI(
    title="Multi-Agent Orchestration Aggregator V1.5.1",
    description="Aggregator with refined chain step execution targeting direct model endpoints.",
    version="1.5.1",
    on_startup=[],
    on_shutdown=[]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists(BASE_WORKSPACE_DIR):
    os.makedirs(BASE_WORKSPACE_DIR, exist_ok=True)
app.mount("/workspaces", StaticFiles(directory=BASE_WORKSPACE_DIR, html=True), name="workspaces")

# --- Agent Discovery and Communication Utilities ---
def is_port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.2) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False
    finally:
        s.close()

def _determine_target_url_and_payload(
    req_data_union: Union[MessageRequest, InteractiveStepRequest, ChainStep, Dict[str, Any]],
    input_prompt_text: str, # Explicitly pass the primary prompt/text
    use_direct_endpoint_if_main_server_model: bool = False
) -> tuple[str, dict, str]:
    """
    Determines target URL and payload for an agent call.
    `input_prompt_text` is the core text for the LLM.
    `use_direct_endpoint_if_main_server_model`: If True, for models on MAIN_AGENT_SERVER_URL,
    targets their specific named endpoint (e.g., /coder, /orchestratorthinking) rather than /ask.
    This is key for chain steps or direct single calls.
    """
    agent_id_obj_data = None # This will be the dict version of AgentIdentifier
    agent_name_fallback = None
    agent_port_fallback = None
    max_tokens_fallback = 10000
    temperature_fallback = None

    if isinstance(req_data_union, dict):
        # Directly use the dict keys
        agent_id_obj_data = req_data_union.get('agentIdentifier') # Note: alias 'agentIdentifier'
        agent_name_fallback = req_data_union.get('agent_name')
        agent_port_fallback = req_data_union.get('agent_port')
        max_tokens_fallback = req_data_union.get('max_tokens', 10000)
        temperature_fallback = req_data_union.get('temperature')
    else: # Pydantic models (MessageRequest, InteractiveStepRequest, ChainStep)
        # Use model_dump to get a dict
        req_data_dict = req_data_union.model_dump(by_alias=True, exclude_none=True)
        agent_id_obj_data = req_data_dict.get('agentIdentifier')
        agent_name_fallback = req_data_dict.get('agent_name')
        agent_port_fallback = req_data_dict.get('agent_port')
        max_tokens_fallback = req_data_dict.get('max_tokens', 10000)
        temperature_fallback = req_data_dict.get('temperature')

    # local_model.py endpoints expect "prompt" key for input text
    payload_dict = {
        "prompt": input_prompt_text,
        "max_tokens": max_tokens_fallback,
    }
    if temperature_fallback is not None:
        payload_dict["temperature"] = temperature_fallback

    if agent_id_obj_data:
        # agent_id_obj_data is now guaranteed to be a dict
        _type = agent_id_obj_data.get('type')
        _name = agent_id_obj_data.get('name')
        _port = agent_id_obj_data.get('port')

        if _name and (_type == "specialist_main_server" or (_type == "orchestrator_main_server" and use_direct_endpoint_if_main_server_model)):
            # If it's a main server model (specialist OR orchestrator model used directly),
            # target its specific named endpoint.
            url_safe_name = _name.lower().replace(" ", "_").replace("-", "_")
            target_url = f"{MAIN_AGENT_SERVER_URL}/{url_safe_name}"
            return target_url, payload_dict, _name
        elif _type == "orchestrator_main_server" and not use_direct_endpoint_if_main_server_model:
            # If it's the orchestrator model AND we don't want a direct endpoint (i.e., want the /ask loop)
            target_url = f"{MAIN_AGENT_SERVER_URL}/ask"
            return target_url, payload_dict, _name or "OrchestratorLoop"
        elif _type == "network_scanned" and _port is not None:
            # Network scanned agents are assumed to have a generic /ask endpoint
            target_url = f"http://localhost:{_port}/ask"
            return target_url, payload_dict, f"Port_{_port}"

    # Fallback to older name/port logic if agent_identifier_obj_data is missing or its type is unhandled
    if agent_name_fallback:
        # Assumes direct named endpoint if only name is provided
        url_safe_name = agent_name_fallback.lower().replace(" ", "_").replace("-", "_")
        target_url = f"{MAIN_AGENT_SERVER_URL}/{url_safe_name}"
        return target_url, payload_dict, agent_name_fallback
    elif agent_port_fallback is not None:
        # If MAIN_AGENT_SERVER_PORT is explicitly targeted by port AND we don't want a direct endpoint
        if agent_port_fallback == MAIN_AGENT_SERVER_PORT and not use_direct_endpoint_if_main_server_model:
            target_url = f"{MAIN_AGENT_SERVER_URL}/ask"
            return target_url, payload_dict, f"OrchestratorLoop_Port_{agent_port_fallback}"
        else: # Other network scanned ports or direct main server port when direct endpoint is forced
            target_url = f"http://localhost:{agent_port_fallback}/ask"
            return target_url, payload_dict, f"Port_{agent_port_fallback}"

    raise ValueError(f"Agent target (port, name, or identifier object) not specified or could not be resolved.")


def call_agent_once_v1_5(
    conversation_id: Optional[str],
    request_details_union: Union[MessageRequest, InteractiveStepRequest, ChainStep, Dict[str, Any]],
    current_prompt_for_agent: str, # The fully constructed prompt text to send
    use_direct_endpoint_if_main_server_model: bool = False # Flag to prefer direct named endpoint on local_model.py
) -> str:
    call_dict_for_payload_determination: Dict[str, Any]

    # Convert Pydantic models to dicts for consistent attribute/key access
    if not isinstance(request_details_union, dict):
        call_dict_for_payload_determination = request_details_union.model_dump(by_alias=True, exclude_none=True)
    else:
        call_dict_for_payload_determination = request_details_union.copy()

    try:
        target_url, payload_dict, agent_id_str = _determine_target_url_and_payload(
            call_dict_for_payload_determination, # Pass the dict version
            input_prompt_text=current_prompt_for_agent,
            use_direct_endpoint_if_main_server_model=use_direct_endpoint_if_main_server_model
        )
    except ValueError as e:
        if conversation_id:
            log_message_db(conversation_id, "system_error", f"Agent target resolution error: {e}", "ResolutionError")
        raise HTTPException(status_code=400, detail=str(e))

    if conversation_id:
        log_message_db(conversation_id, f"aggregator_to_{agent_id_str}_prompt",
                       f"URL: {target_url}\nPayload: {json.dumps(payload_dict, indent=2)[:1000]}...", agent_id_str)
    raw_response_text = ""
    try:
        r = requests.post(target_url, json=payload_dict, timeout=1800)
        raw_response_text = r.text
        r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = r.json()
        # Local_model.py direct endpoints return "answer", /ask returns "final_answer"
        answer = data.get("answer", data.get("final_answer", data.get("response", "")))
        if conversation_id:
            log_message_db(conversation_id, f"agent_response_from_{agent_id_str}", answer, agent_id_str, raw_response_text)
        return answer
    except requests.exceptions.Timeout:
        err_msg = f"Timeout calling agent {agent_id_str} at {target_url}"
        if conversation_id: log_message_db(conversation_id, "system_error", err_msg, agent_id_str, "REQUEST_TIMEOUT")
        raise HTTPException(status_code=504, detail=err_msg)
    except requests.exceptions.RequestException as e:
        err_msg = f"Error calling agent {agent_id_str}: {str(e)}. Raw response: {raw_response_text[:500]}"
        if conversation_id: log_message_db(conversation_id, "system_error", err_msg, agent_id_str, raw_response_text)
        raise HTTPException(status_code=502, detail=err_msg)
    except json.JSONDecodeError as e:
        err_msg = f"Failed to decode JSON from agent {agent_id_str}. Error: {e}. Raw: {raw_response_text[:500]}"
        if conversation_id: log_message_db(conversation_id, "system_error", err_msg, agent_id_str, raw_response_text)
        raise HTTPException(status_code=500, detail=err_msg)


# --- Output Assessment ---
def assess_agent_output_for_directives_and_quality(
    agent_response: str,
    agent_role: str,
    original_instruction_to_agent: str # For context, not used in this heuristic version yet
) -> Dict[str, Any]:
    """
    Assesses agent output for basic quality and extracts potential directives.
    This is a heuristic-based version, an LLM-based assessment could be added later.
    """
    assessment = {
        "status": "UNKNOWN", # e.g., NEEDS_CLARIFICATION, HAS_DIRECTIVES, LOOKS_ACCEPTABLE, POTENTIAL_QUALITY_ISSUE
        "feedback": [],
        "extracted_code_blocks": [], # {"lang": "python", "content": "...", "filename_comment": "main.py"}
        "extracted_commands": [],    # "pytest -v"
        "extracted_file_directives": [], # {"filename": "data.json", "content": "{...}"}
        "is_final_for_sub_task": True # Assume final unless it's clearly asking a question
    }
    response_lower = agent_response.lower()

    # 1. Basic Sanity & Refusal
    if not agent_response.strip():
        assessment["status"] = "POTENTIAL_QUALITY_ISSUE"
        assessment["feedback"].append("Response is empty or only whitespace.")
        assessment["is_final_for_sub_task"] = False # Empty is not final
        return assessment

    common_refusals = ["as an ai language model", "i cannot fulfill", "i am unable to", "my apologies, but i can't"]
    for refusal in common_refusals:
        if refusal in response_lower:
            assessment["status"] = "POTENTIAL_QUALITY_ISSUE"
            assessment["feedback"].append(f"Response contains a common refusal phrase: '{refusal}'.")
            assessment["is_final_for_sub_task"] = True # Refusal is a final statement for that attempt
            return assessment

    # 2. Conversational Check (Is the agent asking a question back?)
    if agent_response.strip().endswith("?") or \
       any(q_phrase in response_lower for q_phrase in ["could you please clarify", "do you mean", "what about", "how should i proceed"]):
        assessment["status"] = "NEEDS_CLARIFICATION"
        assessment["feedback"].append("Agent seems to be asking a clarifying question.")
        assessment["is_final_for_sub_task"] = False
        # No need to look for directives if it's a question
        return assessment

    # 3. Directive Extraction (if not clearly conversational)
    # Code Blocks (```lang // filename.ext \n code ```)
    # Regex to capture language (optional), filename from comment (optional), and code
    code_block_pattern = re.compile(r"```(\w*)?\s*(?://\s*([\w./-]+)|\#\s*([\w./-]+))?\s*\n([\s\S]*?)\n```")
    for match in code_block_pattern.finditer(agent_response):
        lang = (match.group(1) or "").strip()
        filename_comment = (match.group(2) or match.group(3) or "").strip()
        code_content = match.group(4).strip()
        assessment["extracted_code_blocks"].append({
            "lang": lang,
            "filename_comment": filename_comment,
            "content": code_content
        })

    # TERMINAL_CMD:
    term_cmd_pattern = re.compile(r"^TERMINAL_CMD:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
    for match in term_cmd_pattern.finditer(agent_response):
        assessment["extracted_commands"].append(match.group(1).strip())

    # FILE: path/to/filename.ext \n content...
    file_directive_pattern = re.compile(r"^FILE:\s*([\w./-]+)\s*\n([\s\S]*?)(?=(?:^FILE:\s*)|$)", re.MULTILINE)
    # Extracting the content part until the next FILE: or end of string
    current_pos = 0
    for match in file_directive_pattern.finditer(agent_response):
        assessment["extracted_file_directives"].append({
            "filename": match.group(1).strip(),
            "content": match.group(2).strip() # Content until next FILE or end of response
        })

    # 4. Role-Specific Heuristics & Status Update
    has_any_directive = bool(assessment["extracted_code_blocks"] or \
                             assessment["extracted_commands"] or \
                             assessment["extracted_file_directives"])

    if has_any_directive:
        assessment["status"] = "HAS_DIRECTIVES"
        assessment["feedback"].append("Directives (code, command, or file) found.")
    else:
        # If no directives and not a question, it might be a textual answer or plan
        assessment["status"] = "LOOKS_ACCEPTABLE" # Default for textual responses
        assessment["feedback"].append("Response appears to be textual or a plan; no specific directives extracted.")

    if "developer" in agent_role.lower() or "coder" in agent_role.lower():
        if not assessment["extracted_code_blocks"] and not assessment["extracted_file_directives"]:
            assessment["feedback"].append(f"Note: Developer role '{agent_role}' did not produce explicit code blocks or FILE directives in this turn.")
            if assessment["status"] == "LOOKS_ACCEPTABLE": # If it was otherwise acceptable text
                assessment["status"] = "POTENTIAL_QUALITY_ISSUE" # More of a warning now
    elif "tester" in agent_role.lower():
        if not assessment["extracted_commands"] and not any(kw in response_lower for kw in ["pass", "fail", "passed", "failed"]):
            assessment["feedback"].append(f"Note: Tester role '{agent_role}' did not run commands or state a clear PASS/FAIL.")
            if assessment["status"] == "LOOKS_ACCEPTABLE":
                 assessment["status"] = "POTENTIAL_QUALITY_ISSUE"

    # 5. Placeholder content
    placeholders = ["// TODO", "# TODO", "[insert", "(your code here)"]
    for ph in placeholders:
        if ph.lower() in response_lower:
            assessment["status"] = "POTENTIAL_QUALITY_ISSUE"
            assessment["feedback"].append(f"Response may contain placeholder content: '{ph}'.")
            break

    if not assessment["feedback"]:
        assessment["feedback"].append("Assessment complete.")

    return assessment

# --- ENDPOINTS ---
# MODIFIED: get_agents_v1_5 to populate the global _discovered_local_models cache
@app.get("/agents")
def get_agents_v1_5():
    global _discovered_local_models # Declare global to modify the cached list
    active_agents_list: List[Dict[str, Any]] = []
    # 1. Scan local ports (excluding main model server port)
    for port in range(AGENT_SCAN_PORTS_START, AGENT_SCAN_PORTS_END + 1):
        if MAIN_AGENT_SERVER_PORT and port == MAIN_AGENT_SERVER_PORT:
            continue
        if is_port_open(port):
            active_agents_list.append({
                "port": port,
                "type": "network_scanned",
                "name": f"Agent_Port_{port}",
                "role_description": "Generic agent discovered on local network via port scan."
            })

    # 2. Query main model server (local_model.py) for its configured agents
    if MAIN_AGENT_SERVER_URL:
        try:
            response = requests.get(f"{MAIN_AGENT_SERVER_URL}/list_main_agents", timeout=5)
            response.raise_for_status()
            main_server_agents_data = response.json()
            _discovered_local_models = [] # Clear and repopulate the global cache
            if isinstance(main_server_agents_data, list):
                for agent_data in main_server_agents_data:
                    _discovered_local_models.append(agent_data) # Store the raw dict here
                    agent_type = agent_data.get("type", "specialist_main_server") # local_model.py now provides type directly

                    active_agents_list.append({
                        "name": agent_data.get("name"),
                        "type": agent_type,
                        "role_description": agent_data.get("role_description"),
                        "n_ctx": agent_data.get("n_ctx"),
                        "loading_strategy": agent_data.get("loading_strategy"),
                        "port": None # Port is not relevant for these, accessed by name
                    })
            else:
                log_message_db("GLOBAL_SYSTEM_LOG", "error", f"Unexpected format from {MAIN_AGENT_SERVER_URL}/list_main_agents: {str(main_server_agents_data)[:200]}")

        except requests.exceptions.RequestException as e:
            msg = f"Warning: Could not fetch agents from main server {MAIN_AGENT_SERVER_URL}: {e}"
            print(msg)
            log_message_db("GLOBAL_SYSTEM_LOG", "error", msg)
            _discovered_local_models = [] # Clear cache on error
    return {"active_agents": active_agents_list}


# MODIFIED: send_message_v1_5 - Intercept frontend summarization calls
@app.post("/message")
def send_message_v1_5(req: MessageRequest):
    start_time = time.time()
    conversation_id_for_log = req.conversation_id or "NO_CONVO_ID_MESSAGE_ENDPOINT"

    # Determine agent for logging purposes
    agent_log_identifier = "UnknownAgent"
    if req.agent_identifier_obj:
        agent_log_identifier = req.agent_identifier_obj.name or f"AgentObjType_{req.agent_identifier_obj.type}"
        if req.agent_identifier_obj.port:
            agent_log_identifier += f"_Port{req.agent_identifier_obj.port}"
    elif req.agent_name:
        agent_log_identifier = req.agent_name
    elif req.agent_port is not None:
        agent_log_identifier = f"Port_{req.agent_port}"

    log_message_db(
        conversation_id_for_log,
        "system_api_call_received",
        f"Received POST /message request to agent: {agent_log_identifier}. Question snippet: '{req.question[:100]}...'",
        agent_log_identifier
    )

    if not req.question or not req.question.strip():
        log_message_db(
            conversation_id_for_log,
            "system_error_apicall",
            "Validation Error: Question cannot be empty for /message endpoint.",
            agent_log_identifier
        )
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if req.conversation_id:
        log_message_db(
            req.conversation_id,
            f"user_direct_message_to_{agent_log_identifier}",
            req.question,
            agent_log_identifier
        )

    # --- NEW LOGIC: Intercept summarization requests from the frontend ---
    # The frontend's summarizeConversation function generates a specific prompt.
    # We detect this prompt using a unique keyword.
    SUMMARY_PROMPT_KEYWORD = "Please provide a concise summary of the following interaction"
    if SUMMARY_PROMPT_KEYWORD in req.question:
        log_message_db(
            conversation_id_for_log,
            "system_agent_override",
            f"Intercepted summarization request from frontend. Routing to internal summarization logic (PerformAction model).",
            "FrontendSummarizationInterceptor"
        )
        # Call the dedicated internal summarization function directly.
        # It already handles model selection (PerformAction) and logging to DB.
        summary_result = summarize_conversation_segment(
            conversation_id=req.conversation_id,
            text_to_summarize=req.question, # The entire question is the text to summarize
            step_label="Frontend_Summary_Call" # Log label for this pathway
        )
        # Return the summary. The frontend expects an {"answer": "..."} structure.
        return {"answer": summary_result}
    # --- END NEW LOGIC ---

    # Original logic for non-summarization /message calls:
    # If it's not a summarization request, proceed with the agent identified in the request.
    final_agent_identifier_obj_for_call = req.agent_identifier_obj
    final_prompt_to_send = req.question # Use the original prompt

    # If the original agent_identifier_obj was None, try to reconstruct from name/port
    if final_agent_identifier_obj_for_call is None:
        if req.agent_name:
            # Attempt to find the full definition from the global cache
            found_def_dict = next((m for m in _discovered_local_models if m.get("name") == req.agent_name), None)
            if found_def_dict:
                final_agent_identifier_obj_for_call = AgentIdentifier(**found_def_dict)
            else:
                # Fallback if name provided but not found in known main server models
                final_agent_identifier_obj_for_call = AgentIdentifier(name=req.agent_name, type="specialist_main_server")
                log_message_db(conversation_id_for_log, "system_warning", f"Agent name '{req.agent_name}' not found in known main server models; using generic identifier for call.", "AgentResolution")
        elif req.agent_port is not None:
            final_agent_identifier_obj_for_call = AgentIdentifier(port=req.agent_port, type="network_scanned")
        else:
            raise HTTPException(status_code=400, detail="No valid agent identifier (object, name, or port) provided in request.")

    # Prepare the payload for call_agent_once_v1_5 using the determined agent and original prompt/settings
    call_payload_for_agent_once = {
        "agentIdentifier": final_agent_identifier_obj_for_call.model_dump(exclude_none=True),
        "question": final_prompt_to_send,
        "max_tokens": req.max_tokens,
        "conversation_id": req.conversation_id,
        "temperature": req.temperature # Pass the temperature from the original request
    }

    try:
        log_message_db(
            conversation_id_for_log,
            "system_internal_call_attempt",
            f"Final call target for /message (non-summarization) is agent '{final_agent_identifier_obj_for_call.name or final_agent_identifier_obj_for_call.port}'.",
            final_agent_identifier_obj_for_call.name or str(final_agent_identifier_obj_for_call.port)
        )

        answer = call_agent_once_v1_5(
            conversation_id=req.conversation_id,
            request_details_union=call_payload_for_agent_once, # Pass the constructed dict
            current_prompt_for_agent=final_prompt_to_send, # The prompt to use for the LLM
            use_direct_endpoint_if_main_server_model=True # Crucial: for /message, usually prefer direct model endpoints
        )

        end_time = time.time()
        log_message_db(
            conversation_id_for_log,
            "system_api_call_success",
            f"Successfully processed POST /message to agent: {final_agent_identifier_obj_for_call.name or final_agent_identifier_obj_for_call.port}. Duration: {end_time - start_time:.2f}s. Answer snippet: '{str(answer)[:100]}...'",
            final_agent_identifier_obj_for_call.name or str(final_agent_identifier_obj_for_call.port)
        )
        return {"answer": answer}

    except HTTPException as http_exc:
        end_time = time.time()
        error_detail_log = f"HTTPException from agent call: Status={http_exc.status_code}, Detail='{http_exc.detail}'. Duration: {end_time - start_time:.2f}s."
        log_message_db(
            conversation_id_for_log,
            "system_error_agent_call_http",
            error_detail_log,
            final_agent_identifier_obj_for_call.name or str(final_agent_identifier_obj_for_call.port),
            raw_response=json.dumps({"status_code": http_exc.status_code, "detail": http_exc.detail})
        )
        print(f"ERROR in /message for agent {final_agent_identifier_obj_for_call.name or final_agent_identifier_obj_for_call.port}: {error_detail_log}")
        raise http_exc

    except ValueError as ve:
        end_time = time.time()
        error_detail_log = f"ValueError during /message processing for agent '{final_agent_identifier_obj_for_call.name or final_agent_identifier_obj_for_call.port}': {str(ve)}. Duration: {end_time - start_time:.2f}s."
        log_message_db(
            conversation_id_for_log,
            "system_error_apicall_validation",
            error_detail_log,
            final_agent_identifier_obj_for_call.name or str(final_agent_identifier_obj_for_call.port),
            raw_response=str(ve)
        )
        print(f"ERROR in /message (ValueError) for agent {final_agent_identifier_obj_for_call.name or final_agent_identifier_obj_for_call.port}: {error_detail_log}")
        raise HTTPException(status_code=400, detail=f"Bad Request: {str(ve)}")

    except Exception as e:
        end_time = time.time()
        error_detail_log = f"Unexpected internal server error in /message while contacting agent '{final_agent_identifier_obj_for_call.name or final_agent_identifier_obj_for_call.port}': {str(e)}. Duration: {end_time - start_time:.2f}s."
        log_message_db(
            conversation_id_for_log,
            "system_error_apicall_unhandled",
            error_detail_log,
            final_agent_identifier_obj_for_call.name or str(final_agent_identifier_obj_for_call.port),
            raw_response=str(e)
        )
        print(f"CRITICAL ERROR in /message for agent {final_agent_identifier_obj_for_call.name or final_agent_identifier_obj_for_call.port}: {error_detail_log}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: An unexpected error occurred processing your request with agent '{final_agent_identifier_obj_for_call.name or final_agent_identifier_obj_for_call.port}'.")


# --- Command Execution and File System Utilities (Enhanced) ---
def create_merged_env_for_project(conversation_id: str) -> Dict[str, str]:
    env_copy = os.environ.copy()
    try:
        # Fetch project data to get "env" configuration
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT project_json FROM projects WHERE conversation_id = ?", (conversation_id,))
        row = cur.fetchone()
        conn.close()
        if row and row[0]:
            project_data = json.loads(row[0])
            project_env_config = project_data.get("env")
            if project_env_config and isinstance(project_env_config, dict):
                for k, v_val in project_env_config.items():
                    env_copy[k] = str(v_val)
    except Exception as e:
        log_message_db(conversation_id, "system_warning", f"Could not load project env for command execution: {e}. Using system env.")
    return env_copy

def run_command_and_log(cmd: str, conversation_id: str, timeout_seconds: int = 180) -> tuple[str, str, int]:
    """ Robust command execution, logging, and CWD management. """
    start_time = time.time()
    convo_folder = get_conversation_folder(conversation_id) # Ensures it exists
    original_cwd = os.getcwd()
    blocked_patterns = [r"rm\s+-rf", r"mkfs", r"shutdown", r"reboot", r"sudo\s+", r":\s*\(\s*\)\s*\{.*&?\s*\}\s*;", r">\s*/dev/sd[a-z]"]
    stdout_res, stderr_res, return_code_res = "", "", -1
    process = None # Initialize process to None

    try:
        for pattern in blocked_patterns:
            if re.search(pattern, cmd.lower()):
                msg = f"Command '{cmd}' blocked due to security policy (matched: '{pattern}')."
                log_message_db(conversation_id, "system_command_blocked", msg, agent_identifier="SecurityPolicy")
                store_command_execution_db(conversation_id, cmd, start_time, time.time(), "", msg, -99)
                return "", msg, -99

        final_cmd_to_run = cmd
        if IS_WINDOWS: # Adjust common commands for Windows
            if cmd.startswith("npm "): final_cmd_to_run = cmd.replace("npm ", "npm.cmd ", 1)
            elif cmd.startswith("npx "): final_cmd_to_run = cmd.replace("npx ", "npx.cmd ", 1)
            elif cmd.startswith("python "): pass # Usually OK
            # Add other windows specific adjustments if needed

        log_message_db(conversation_id, "system_command_executing", f"Executing: {final_cmd_to_run} in {convo_folder}", agent_identifier="TerminalInterface")
        merged_env = create_merged_env_for_project(conversation_id)

        os.chdir(convo_folder)
        process = subprocess.Popen(final_cmd_to_run, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=merged_env, cwd=convo_folder)
        try:
            stdout_res, stderr_res = process.communicate(timeout=timeout_seconds)
            return_code_res = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            stdout_res_temp, stderr_res_temp = process.communicate() # Get any output before kill
            stdout_res = stdout_res_temp or ""
            stderr_res = (stderr_res_temp or "") + f"\n[AGGREGATOR_INFO]: Process timed out after {timeout_seconds} seconds and was killed."
            return_code_res = -98 # Specific code for timeout
            log_message_db(conversation_id, "system_command_timeout", f"Command timed out: {final_cmd_to_run}", agent_identifier="TerminalInterface")
    except Exception as exc:
        end_time = time.time() # Capture end time even on exception before Popen
        stderr_res += f"\n[AGGREGATOR_EXCEPTION_PRE_EXEC]: {str(exc)}"
        return_code_res = -97 # Specific code for aggregator exception during setup
        log_message_db(conversation_id, "system_command_error", f"Error preparing to execute '{cmd}': {exc}", raw_response=str(exc), agent_identifier="TerminalInterface")
    finally:
        end_time = time.time() # Ensure end_time is always set
        os.chdir(original_cwd) # Always change back CWD

    # Store execution result, regardless of Popen success/failure (if Popen was reached)
    if process or return_code_res in [-97, -99]: # Check if Popen was actually initiated, or if it was a pre-exec error/block
        store_command_execution_db(conversation_id, final_cmd_to_run if process else cmd, start_time, end_time, stdout_res, stderr_res, return_code_res)
        if process: # Only log command output if process was actually run
            log_content = (
                f"CMD_RAN: {final_cmd_to_run}\nRC: {return_code_res}\n"
                f"STDOUT_SNIPPET:\n{stdout_res[:1000]}{'...' if len(stdout_res)>1000 else ''}\n"
                f"STDERR_SNIPPET:\n{stderr_res[:1000]}{'...' if len(stderr_res)>1000 else ''}"
            )
            log_message_db(conversation_id, "system_command_output", log_content, raw_response=f"STDOUT_FULL: {stdout_res}\nSTDERR_FULL: {stderr_res}", agent_identifier="TerminalInterface")

            if return_code_res != 0:
                repeated_fail_msg = _check_repeated_command_fails_db(conversation_id, final_cmd_to_run)
                if repeated_fail_msg:
                    log_message_db(conversation_id, "system_command_warning", repeated_fail_msg, agent_identifier="SupervisorLogic")
    else: # Should not happen if Popen was not reached and not blocked/pre-exec error
        store_command_execution_db(conversation_id, cmd, start_time, time.time(), "", "Process Popen not initiated due to unknown reason.", -100)

    return stdout_res, stderr_res, return_code_res

def store_command_execution_db(convo_id: str, cmd_str: str, start_t: float, end_t: float,
                            stdout_val: str, stderr_val: str, rc_val: int):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO executed_commands (
            conversation_id, command, start_time, end_time, stdout, stderr, return_code, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (convo_id, cmd_str, start_t, end_t, stdout_val, stderr_val, rc_val, time.time()))
    conn.commit()
    conn.close()

def _check_repeated_command_fails_db(conversation_id: str, cmd_str: str, threshold: int = 3) -> Optional[str]:
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        SELECT return_code FROM executed_commands
        WHERE conversation_id=? AND command=?
        ORDER BY id DESC LIMIT ?
    """, (conversation_id, cmd_str, threshold))
    rows = cur.fetchall()
    conn.close()
    if len(rows) < threshold: return None
    fail_count = sum(1 for row_rc in rows if row_rc[0] != 0)
    if fail_count >= threshold:
        return f"Supervisor Note: Command '{cmd_str[:100]}...' has failed {fail_count} times in the last {threshold} attempts. Consider review."
    return None

def determineTypeFromFilename(filename: str, lang_hint: Optional[str] = None) -> str:
    """Determine a generic type for a workspace item based on filename or language hint."""
    name_lower = filename.lower()
    lang_lower = (lang_hint or "").lower()

    # Priority 1: Specific extensions/languages
    if name_lower.endswith((".py", ".pyw")) or lang_lower == "python": return "code"
    if name_lower.endswith((".js", ".jsx", ".ts", ".tsx")) or lang_lower in ["javascript", "typescript", "jsx", "tsx"]: return "code"
    if name_lower.endswith((".html", ".htm")) or lang_lower in ["html", "htmlmixed"]: return "html"
    if name_lower.endswith(".css") or lang_lower == "css": return "css"
    if name_lower.endswith(".sh") or lang_lower in ["shell", "bash", "sh"]: return "code" # shell scripts are code
    if name_lower.endswith((".md", ".markdown")): return "text" # Markdown treated as text
    if name_lower.endswith(".json") or lang_lower == "json": return "text" # JSON usually edited as text

    # Priority 2: Generic language hint (if not "text" or "plaintext")
    if lang_lower and lang_lower not in ["text", "plaintext"]: return "code"

    return "text" # Default for unknown types

def _write_code_blocks_to_workspace(conversation_id: str, code_blocks: List[Dict[str, Any]], source_description: str) -> List[str]:
    """Writes code blocks to workspace files. Handles filename comments."""
    conversation_folder = get_conversation_folder(conversation_id)
    files_changed = []
    for cb_idx, cb_data in enumerate(code_blocks):
        content = cb_data.get("content", "")
        filename_comment = cb_data.get("filename_comment", "")
        lang = cb_data.get("lang", "txt")

        filename = filename_comment # Use comment first
        if not filename: # Fallback if no filename comment
            filename = f"generated_code_{source_description.replace(' ','_')}_{cb_idx+1}.{lang or 'txt'}"
        
        # Basic sanitization
        safe_filename = re.sub(r'[^\w.\-_/]', '_', filename)
        if not safe_filename: # If sanitization results in empty
            safe_filename = f"untitled_code_{int(time.time())}_{cb_idx}.{lang or 'txt'}"
        
        # Ensure filename is relative, not absolute, and prevent directory traversal
        # Use os.path.normpath and lstrip(os.sep) to correctly handle paths on all OS
        safe_filename = os.path.normpath(safe_filename).lstrip(os.sep)
        if ".." in safe_filename:
            log_message_db(conversation_id, "system_error", f"Skipping potentially unsafe filename from {source_description}: '{filename}' -> '{safe_filename}'")
            continue

        file_path = os.path.join(conversation_folder, safe_filename)
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True) # Ensure parent dirs exist
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            log_message_db(conversation_id, "system_file_write", f"Code from {source_description} written to '{safe_filename}'.", agent_identifier=source_description)
            files_changed.append(safe_filename)
        except Exception as e:
            log_message_db(conversation_id, "system_error", f"File Write Error for '{safe_filename}' (from {source_description}): {str(e)}", agent_identifier=source_description)
    

    if files_changed:
        _save_current_disk_state_to_db(conversation_id)


    return files_changed

def run_terminal_commands_from_text(agent_text: str, conversation_id: str, agent_identifier_for_log: str):
    """Parses TERMINAL_CMD and shell blocks, then runs them."""
    commands_to_run = []
    # TERMINAL_CMD:
    term_cmd_pattern = re.compile(r"^TERMINAL_CMD:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
    for match in term_cmd_pattern.finditer(agent_text):
        commands_to_run.append(match.group(1).strip())

    # Shell blocks (```bash/shell/sh ... ```)
    shell_block_pattern = re.compile(r"```(?:bash|shell|sh)\s*\n([\s\S]*?)\n```", re.IGNORECASE)
    for match in shell_block_pattern.finditer(agent_text):
        code_block = match.group(1).strip()
        for line in code_block.splitlines():
            cmd_line = line.strip()
            if cmd_line and not cmd_line.startswith("#"): # Basic comment skipping
                commands_to_run.append(cmd_line)
    
    if commands_to_run:
        log_message_db(conversation_id, "system_directive_parse", f"Identified commands from {agent_identifier_for_log} to run: {commands_to_run}")
        for cmd in commands_to_run:
            run_command_and_log(cmd, conversation_id)

def advanced_parse_and_handle_file_directives(agent_text: str, conversation_id: str, agent_role_or_id: str) -> List[str]:
    """Parses FILE: directives and writes content to workspace. Returns list of files changed."""
    file_directive_pattern = re.compile(r"^FILE:\s*([\w./-]+)\s*\n([\s\S]*?)(?=(?:^FILE:\s*)|$)", re.MULTILINE)
    file_ops_to_perform = [] # Store {"filename": ..., "content": ...}
    changed_files_list = []

    for match in file_directive_pattern.finditer(agent_text):
        filename = match.group(1).strip()
        content = match.group(2).strip()
        if filename and content is not None: # Allow empty content for a file
            file_ops_to_perform.append({"title": filename, "content": content}) # Use "title" to match _write_code_blocks
            log_message_db(conversation_id, "system_directive_parse", f"Parsed FILE directive for '{filename}' from {agent_role_or_id}.")
        else:
            log_message_db(conversation_id, "system_warning", f"Ignored FILE directive from {agent_role_or_id} with missing filename or content (if content was expected but not found).")

    if file_ops_to_perform:
        changed_files_list = _write_code_blocks_to_workspace(conversation_id, file_ops_to_perform, f"FILE_Directives_from_{agent_role_or_id}")
    return changed_files_list


# --- Context Database (Memory) Storage ---
def _store_summary_to_db(conversation_id: str, summary_content: str, agent_name_used: str, step_label: str):
    """
    Internal helper to store a summary in the memories table with detailed logging.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        now = time.time()
        # Sanitize agent_name_used for use in a key (replace common problematic chars)
        sanitized_agent_name = agent_name_used.replace(' ', '_').replace('/', '_').replace('.', '_').replace(':', '_')
        summary_key = f"summary_{step_label}_{int(now)}_{sanitized_agent_name}"

        log_message_db(conversation_id, "system_memory_store_attempt",
                       f"Attempting to store summary for step '{step_label}' (key: '{summary_key}') generated by '{agent_name_used}'.",
                       agent_identifier="MemoryManager")

        cur.execute("""
            INSERT INTO memories (conversation_id, key, value, timestamp)
            VALUES (?, ?, ?, ?)
        """, (conversation_id, summary_key, summary_content, now))
        conn.commit()

        log_message_db(conversation_id, "system_memory_store_success",
                       f"Successfully stored summary for step '{step_label}' with key '{summary_key}'.",
                       agent_identifier="MemoryManager")

    except sqlite3.Error as sql_e:
        error_msg = f"SQLite error storing summary for step '{step_label}' (agent: {agent_name_used}): {sql_e}"
        print(f"AGGREGATOR ERROR: {error_msg}")
        log_message_db(conversation_id, "system_memory_store_failure", error_msg, agent_identifier="MemoryManager")
    except Exception as e:
        error_msg = f"Unexpected error storing summary for step '{step_label}' (agent: {agent_name_used}): {e}"
        print(f"AGGREGATOR ERROR: {error_msg}")
        log_message_db(conversation_id, "system_memory_store_failure", error_msg, agent_identifier="MemoryManager")
    finally:
        if conn:
            conn.close()

# MODIFIED: summarize_conversation_segment - now hardcoded to use PerformAction
def summarize_conversation_segment(
    conversation_id: str,
    text_to_summarize: str,
    primary_summarizer_agent_obj: Optional[Any] = None, # This parameter is now IGNORED for model selection
    step_label: str = "UnnamedStep"
) -> str:
    start_time = time.time()
    log_message_db(conversation_id, "system_summary_process_start",
                   f"Starting summarization for step '{step_label}'. Text length: {len(text_to_summarize)} chars.",
                   agent_identifier="SummarizerMain")

    # --- REQUIRED CHANGE: HARDCODE to "PerformAction" as per requirement ---
    summarizer_name = "PerformAction"
    # Look up the model definition from the global cache
    perform_action_model_dict = next((m for m in _discovered_local_models if m.get("name") == summarizer_name), None)

    if not perform_action_model_dict:
        error_msg = f"Error: Summarizer model '{summarizer_name}' not found in the globally discovered model list. Cannot summarize."
        log_message_db(conversation_id, "system_summary_process_error", error_msg, agent_identifier="SummarizerMain")
        print(f"AGGREGATOR ERROR: {error_msg}")
        return f"[Error: {error_msg}]"

    # Construct an AgentIdentifier object from the discovered 'PerformAction' model's details
    summarizer_agent_obj_for_call = AgentIdentifier(
        name=perform_action_model_dict.get("name"),
        port=perform_action_model_dict.get("port"), # This will be None for main server models
        type=perform_action_model_dict.get("type"), # e.g., "specialist_main_server"
        role_description=perform_action_model_dict.get("role_description"),
        n_ctx=perform_action_model_dict.get("n_ctx"),
        loading_strategy=perform_action_model_dict.get("loading_strategy")
    )

    summary_prompt_content = (
        "SYSTEM: You are a summarization expert. Your task is to provide a concise summary of the following text, "
        "focusing on key outcomes, decisions, files created/modified, and crucial information. "
        "This summary will be used to provide context to subsequent agents or for overall project tracking. "
        "Be concise yet comprehensive. Output only the summary, preferably as bullet points."
        f"\n\nTEXT_TO_SUMMARIZE:\n---\n{text_to_summarize}\n---\n\nCONCISE_SUMMARY (bullet points):"
    )

    try:
        # Prepare the payload for call_agent_once_v1_5.
        # This effectively creates a temporary `MessageRequest`-like structure for the internal call.
        temp_message_request_payload = {
            "agentIdentifier": summarizer_agent_obj_for_call.model_dump(exclude_none=True),
            "question": summary_prompt_content,
            "max_tokens": 512,  # Fixed max tokens for summary generation
            "temperature": 0.3  # Fixed temperature for consistent summary generation
        }

        summary_text = call_agent_once_v1_5(
            conversation_id=conversation_id,
            request_details_union=temp_message_request_payload, # Pass the constructed dict
            current_prompt_for_agent=summary_prompt_content,
            use_direct_endpoint_if_main_server_model=True # Ensure direct model endpoint call
        )

        if summary_text and summary_text.strip():
            _store_summary_to_db(conversation_id, summary_text.strip(), summarizer_name, step_label)
            log_message_db(conversation_id, "system_summary_success",
                           f"Successfully summarized by '{summarizer_name}'.",
                           agent_identifier=summarizer_name)
            return summary_text.strip()
        else:
            error_msg = f"Agent '{summarizer_name}' returned an empty or whitespace-only summary."
            log_message_db(conversation_id, "system_summary_empty_response", error_msg, agent_identifier=summarizer_name)
            print(f"AGGREGATOR WARNING: {error_msg}")
            return f"[Error: {error_msg}]"

    except Exception as e:
        error_msg = (
            f"Exception during summarization call with '{summarizer_name}': {repr(e)}"
        )
        print(f"AGGREGATOR ERROR: Summarizer - {error_msg}")
        log_message_db(conversation_id, "system_summary_call_exception", error_msg, agent_identifier=summarizer_name)
        return f"[Error: {error_msg}]"

def get_recent_summaries(conversation_id: str, count: int = 3) -> str:
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    # Get latest `count` summaries, order by timestamp for chronological presentation
    cur.execute("""
        SELECT key, value FROM memories
        WHERE conversation_id = ? AND key LIKE 'summary_%'
        ORDER BY timestamp DESC LIMIT ? 
    """, (conversation_id, count))
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return "No prior summaries available for this conversation."
    # Reverse to get them in chronological order (oldest of the recent first)
    combined = [f"Summary from '{row[0]}':\n{row[1].strip()}" for row in reversed(rows)]
    return "\n\n---\n\n".join(combined)


# --- Interactive Step and Chain Logic ---
@app.post("/interactive_step")
def interactive_step_v1_5(req: InteractiveStepRequest):
    log_identifier = req.agent_identifier_obj.name if req.agent_identifier_obj and req.agent_identifier_obj.name \
                     else str(req.agent_port or "UnknownAgent_Interactive")

    if req.user_message: # Log user's message to this agent
        log_message_db(req.conversation_id, "user_to_interactive_agent", req.user_message, log_identifier)

    if req.finish_step:
        log_message_db(req.conversation_id, "system_interactive_control", "Step finalized by user.", log_identifier)
        return {"done": True, "answer": "Step finalized by user."}

    prior_summaries = get_recent_summaries(req.conversation_id)

    # Build the prompt for a direct, single turn call to the selected agent
    prompt_parts = [
        f"SYSTEM_ROLE_DEFINITION: You are an AI assistant. Your current designated role is: {req.role}",
        f"ROLE_DESCRIPTION: {req.role_description or 'Act according to your role title for this step.'}",
        "\nCONTEXT_FROM_PROJECT_SUMMARIES (Previous Steps/Overall Goal):\n---\n" + prior_summaries + "\n---",
    ]
    if req.last_agent_message:
        prompt_parts.append(f"\nMESSAGE_FROM_PREVIOUS_AGENT_IN_THIS_STEP_IF_ANY:\n{req.last_agent_message}")
    if req.user_message:
        prompt_parts.append(f"\nCURRENT_USER_INSTRUCTION_FOR_THIS_SPECIFIC_STEP:\n{req.user_message}")
    else:
        prompt_parts.append("\nSYSTEM_NOTE_TO_AGENT: No specific user message for this turn; proceed based on your role, prior summaries, and any previous agent message in this step to achieve the step's objective.")

    if req.workspaces:
        prompt_parts.append("\nAVAILABLE_WORKSPACE_FILES (Content previews are truncated):")
        for idx, ws in enumerate(req.workspaces[:5]): # Limit to 5 for prompt brevity
            ws_content_lines = ws.content.splitlines()
            preview = "\n".join(ws_content_lines[:10]) # Preview first 10 lines
            if len(ws_content_lines) > 10: preview += "\n..."
            prompt_parts.append(f"- File {idx+1}: {ws.title} (Type: {ws.type}, Lines: {len(ws_content_lines)})\n  Preview Snippet:\n  ```\n{preview}\n  ```")
        if len(req.workspaces) > 5:
            prompt_parts.append("- ... (and more files exist in the workspace if you need to list or access them).")
    else:
        prompt_parts.append("No workspace files currently available or provided for this step.")
    
    prompt_parts.append(f"\nYOUR_RESPONSE (as {req.role.upper()} for this step):")
    full_prompt_for_agent = "\n\n".join(prompt_parts)

    try:
        # For interactive_step, we always want to hit the direct endpoint of the selected model.
        # This allows the UI to directly interact with any model (including OrchestratorThinking as a simple thinker).
        agent_raw_answer = call_agent_once_v1_5(
            req.conversation_id,
            req,
            full_prompt_for_agent,
            use_direct_endpoint_if_main_server_model=True
        )

        assessment_result = assess_agent_output_for_directives_and_quality(agent_raw_answer, req.role, req.user_message or "")
        log_message_db(req.conversation_id, f"assessment_of_{log_identifier}", json.dumps(assessment_result, indent=2), log_identifier)

        files_changed_by_directives = []
        if assessment_result["status"] not in ["NEEDS_CLARIFICATION", "ERROR"]:
            # Process directives if the agent is not asking a question back
            run_terminal_commands_from_text(agent_raw_answer, req.conversation_id, log_identifier)
            files_changed_by_directives = advanced_parse_and_handle_file_directives(agent_raw_answer, req.conversation_id, log_identifier)

            # Handle code blocks extracted by assessment (if any, and not already handled by FILE:)
            unhandled_code_blocks = []
            for cb in assessment_result["extracted_code_blocks"]:
                # Ensure all necessary keys are present before checking or processing
                if all(key in cb for key in ["content", "filename_comment", "lang"]):
                    is_handled_by_file_directive = any(f_op["filename"] == cb["filename_comment"] for f_op in assessment_result["extracted_file_directives"] if cb["filename_comment"])
                    if cb["content"] and cb["filename_comment"] and not is_handled_by_file_directive:
                        unhandled_code_blocks.append({"title": cb["filename_comment"], "content": cb["content"], "lang": cb["lang"]})
            
            if unhandled_code_blocks:
                newly_changed_from_codeblocks = _write_code_blocks_to_workspace(req.conversation_id, unhandled_code_blocks, f"CodeBlocks_from_{log_identifier}")
                files_changed_by_directives.extend(newly_changed_from_codeblocks)

        # Auto-run new/modified code files (if any files were written by directives or code blocks)
        if files_changed_by_directives:
            auto_run_new_code_files(req.conversation_id, files_changed_by_directives)

        # Summarize this turn of interaction
        interaction_to_summarize = f"User Instruction: {req.user_message or '(No specific user message this turn)'}\n"
        interaction_to_summarize += f"Previous Agent Msg: {req.last_agent_message or '(None)'}\n"
        interaction_to_summarize += f"Agent ({req.role}) Response: {agent_raw_answer}"
        # MODIFIED: Call to summarize_conversation_segment with new signature
        summarize_conversation_segment(
            conversation_id=req.conversation_id,
            text_to_summarize=interaction_to_summarize,
            step_label=f"interactive_step_{req.role.replace(' ','_')}_{log_identifier}"
        )

        return {
            "done": False, # In interactive mode, "done" is controlled by explicit finish_step
            "answer": agent_raw_answer,
            "assessment": assessment_result # Send assessment to FE
        }
    except HTTPException as http_exc: # Re-raise known HTTP exceptions from call_agent_once
        raise http_exc
    except Exception as e:
        error_detail = f"Error during interactive step with {log_identifier}: {str(e)}"
        log_message_db(req.conversation_id, "system_error", error_detail, log_identifier)
        return {"done": False, "answer": f"[SYSTEM_ERROR]: {error_detail}", "assessment": {"status": "ERROR", "feedback": [error_detail]}}


@app.post("/chain")
def chain_agents_v1_5(req: ChainRequest):
    current_context_for_chain = req.initial_prompt
    conversation_id = req.conversation_id or f"chain_auto_{int(time.time())}"
    log_message_db(conversation_id, "user_initial_prompt_for_chain", current_context_for_chain, agent_identifier="ChainInitiator")
    
    # Initialize workspace state for the chain run
    all_workspaces_for_chain: List[WorkspaceData] = []
    try: # Try to load existing workspaces if project exists
        project_data_full = get_project_v1_5(conversation_id)
        loaded_ws_dicts = project_data_full.get("workspaces", [])
        if isinstance(loaded_ws_dicts, list):
            # Ensure each loaded workspace dict is converted to WorkspaceData Pydantic model
            all_workspaces_for_chain = [WorkspaceData(**ws_dict) for ws_dict in loaded_ws_dicts if isinstance(ws_dict, dict)]
    except HTTPException as he:
        if he.status_code != 404: # Project not found is okay, means new chain, but other HTTP errors need logging
            log_message_db(conversation_id, "system_warning", f"Could not load initial workspaces for chain, HTTP error: {he.detail}")
    except Exception as e:
        log_message_db(conversation_id, "system_warning", f"Could not load initial workspaces for chain due to general error: {str(e)}")


    for step_index, step_config in enumerate(req.chain):
        log_id_parts_step = []
        if step_config.agent_identifier_obj: log_id_parts_step.append(step_config.agent_identifier_obj.name or "UnnamedAgentObj")
        if step_config.agent_name: log_id_parts_step.append(step_config.agent_name)
        if step_config.agent_port is not None: log_id_parts_step.append(str(step_config.agent_port))
        log_identifier_step = "_".join(filter(None, log_id_parts_step)) or f"ChainAgent_Step{step_index+1}"
        
        prior_summaries_chain = get_recent_summaries(conversation_id, count=3) # Get up to 3 recent summaries

        # Build prompt for this chain step
        prompt_parts_chain = [
            f"SYSTEM_CHAIN_STEP_DEFINITION: You are part of an automated multi-step chain. This is Step {step_index + 1} of {len(req.chain)}.",
            f"Your designated role for THIS STEP is: {step_config.role}",
            f"Your role description for THIS STEP: {step_config.role_description or 'Act according to your role title for this specific step in the chain.'}",
            "\nCORE_INSTRUCTIONS_FOR_THIS_STEP:",
            "1. Carefully review the `[INPUT_FROM_PREVIOUS_STEP_OR_INITIAL_PROMPT]` which contains the current task or data to process.",
            "2. Consult `[PRIOR_CONTEXT_FROM_SUMMARIES]` for overall project context and previous step outcomes.",
            "3. Review `[ACCESSIBLE_WORKSPACE_FILES]` if any are provided and relevant.",
            "4. Perform your designated task based on your role and the provided input.",
            "5. Your response will be the primary input for the *next* agent in the chain, or the final answer if this is the last step. Ensure it is comprehensive and directly addresses what the next step (or user) needs.",
            "6. To create or modify files in the project workspace, use the format `FILE: path/to/filename.ext` on a new line, followed by the complete file content on subsequent lines. End file content before any other text.",
            "7. To execute terminal commands within the project workspace, use `TERMINAL_CMD: your_command_here` on a new line.",
            "8. If generating code, prefer using `FILE: ...` directives or use filename comments in code blocks (e.g., ```python // my_script.py ... ```) for clarity.",
            f"\nPRIOR_CONTEXT_FROM_SUMMARIES (Overall Project History):\n---\n{prior_summaries_chain}\n---",
        ]
        
        if all_workspaces_for_chain:
            prompt_parts_chain.append("\nACCESSIBLE_WORKSPACE_FILES (Current state. Content snippets are truncated):")
            # Filter out folders for display in prompt, and limit to first 5 for brevity
            filtered_ws_for_prompt = [ws for ws in all_workspaces_for_chain if ws.type != "folder"]
            for idx_ws, ws_data in enumerate(filtered_ws_for_prompt[:5]):
                ws_content_lines_chain = ws_data.content.splitlines()
                preview_chain = "\n".join(ws_content_lines_chain[:10]) + ("\n..." if len(ws_content_lines_chain) > 10 else "")
                prompt_parts_chain.append(f"- File {idx_ws+1}: {ws_data.title} (Type: {ws_data.type}, Lines: {len(ws_content_lines_chain)})\n  Preview Snippet:\n  ```\n{preview_chain}\n  ```")
            if len(filtered_ws_for_prompt) > 5:
                prompt_parts_chain.append("- ... (and more files exist in the workspace).")
        else:
            prompt_parts_chain.append("No workspace files currently exist or are provided for this step.")

        prompt_parts_chain.append(f"\nINPUT_FROM_PREVIOUS_STEP_OR_INITIAL_PROMPT_FOR_STEP_{step_index+1}:\n---\n{current_context_for_chain}\n---")
        prompt_parts_chain.append(f"\nYOUR_RESPONSE_AS_{step_config.role.upper()}_FOR_STEP_{step_index+1}:")
        full_prompt_for_step_agent = "\n\n".join(prompt_parts_chain)

        try:
            # CRITICAL: For chain steps, always use direct model endpoints on local_model.py.
            # This ensures each step hits /coder, /orchestratorthinking, /intent, etc. directly.
            # The step_config Pydantic model is passed to call_agent_once_v1_5.
            agent_raw_answer_step = call_agent_once_v1_5(
                conversation_id,
                step_config, # Pass the Pydantic model instance
                full_prompt_for_step_agent, # This is the actual combined prompt
                use_direct_endpoint_if_main_server_model=True
            )
            
            assessment_step = assess_agent_output_for_directives_and_quality(agent_raw_answer_step, step_config.role or "ChainAgent", current_context_for_chain)
            log_message_db(conversation_id, f"assessment_of_chain_step_{log_identifier_step}", json.dumps(assessment_step, indent=2), log_identifier_step)

            files_changed_this_step = []
            if assessment_step["status"] not in ["NEEDS_CLARIFICATION", "ERROR"]:
                run_terminal_commands_from_text(agent_raw_answer_step, conversation_id, log_identifier_step)
                files_changed_this_step = advanced_parse_and_handle_file_directives(agent_raw_answer_step, conversation_id, log_identifier_step)
                
                # Handle code blocks not explicitly in FILE directives
                unhandled_code_blocks_chain = []
                for cb in assessment_step["extracted_code_blocks"]:
                    # Ensure all necessary keys are present before checking or processing
                    if all(key in cb for key in ["content", "filename_comment", "lang"]):
                        is_handled_by_file_directive = any(f_op["filename"] == cb["filename_comment"] for f_op in assessment_step["extracted_file_directives"] if cb["filename_comment"])
                        if cb["content"] and cb["filename_comment"] and not is_handled_by_file_directive:
                             unhandled_code_blocks_chain.append({"title": cb["filename_comment"], "content": cb["content"], "lang": cb["lang"]})
                
                if unhandled_code_blocks_chain:
                    new_files_codeblocks = _write_code_blocks_to_workspace(conversation_id, unhandled_code_blocks_chain, f"CodeBlocks_from_{log_identifier_step}")
                    files_changed_this_step.extend(new_files_codeblocks)

            # Update all_workspaces_for_chain state if files were changed
            if files_changed_this_step:
                auto_run_new_code_files(conversation_id, files_changed_this_step) # Auto-run if applicable
                
                # Re-read all files from the workspace folder to get the latest state for the next step.
                # This ensures the next agent's prompt includes up-to-date file contents.
                updated_workspaces_after_step = []
                convo_folder_chain = get_conversation_folder(conversation_id)
                for root, _, files_in_dir in os.walk(convo_folder_chain):
                    for file_name in files_in_dir:
                        file_path_full = os.path.join(root, file_name)
                        relative_path = os.path.relpath(file_path_full, convo_folder_chain)
                        try:
                            with open(file_path_full, 'r', encoding='utf-8') as f_read:
                                file_content = f_read.read()
                            file_type_determined = determineTypeFromFilename(relative_path)
                            updated_workspaces_after_step.append(
                                WorkspaceData(title=relative_path, content=file_content, type=file_type_determined)
                            )
                        except Exception as e_file_read:
                            log_message_db(conversation_id, "system_warning", f"Chain: Error reading file '{relative_path}' for workspace update: {e_file_read}")
                all_workspaces_for_chain = updated_workspaces_after_step # Update global state for next loop iteration
                log_message_db(conversation_id, "system_chain_control", f"Workspace state re-read after step {step_index+1} ({len(all_workspaces_for_chain)} files).")

            current_context_for_chain = agent_raw_answer_step # Output of one becomes input to next

            # MODIFIED: Call to summarize_conversation_segment with new signature
            summarize_conversation_segment(
                conversation_id=conversation_id,
                text_to_summarize=f"Input to {step_config.role}: {current_context_for_chain[:500]}...\nOutput from {step_config.role}: {agent_raw_answer_step}",
                step_label=f"chain_step_{step_index+1}_{step_config.role.replace(' ','_')}"
            )
            
        except Exception as e:
            error_msg = f"Error in chain step {step_index + 1} ({log_identifier_step}): {str(e)}"
            log_message_db(conversation_id, "system_chain_error", error_msg, log_identifier_step)
            # If a step in a chain fails, the chain is generally halted.
            raise HTTPException(status_code=500, detail=error_msg)

    log_message_db(conversation_id, "system_chain_final_output", current_context_for_chain, agent_identifier="ChainConclusion")
    return {"final_answer": current_context_for_chain, "conversation_id": conversation_id}


# --- Other Endpoints ---
@app.post("/log_message")
def api_log_message_v1_5(req: LogRequest):
    log_identifier = req.agent_identifier_obj.name if req.agent_identifier_obj and req.agent_identifier_obj.name \
                     else str(req.agent_port or "ClientLog")
    log_message_db(req.conversation_id, req.role, req.content, log_identifier)
    return {"status": "logged"}

@app.get("/conversations")
def list_conversations_v1_5():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    # Get distinct conversation_ids from projects first (likely more recently active)
    cur.execute("SELECT conversation_id, updated_at FROM projects ORDER BY updated_at DESC")
    project_convos = {r[0]: r[1] for r in cur.fetchall()}
    
    # Then from messages for any not in projects
    cur.execute("SELECT conversation_id, MAX(timestamp) AS last_msg_ts FROM messages GROUP BY conversation_id ORDER BY last_msg_ts DESC")
    message_convos = {r[0]: r[1] for r in cur.fetchall()}
    conn.close()

    # Combine and sort: project activity is a strong indicator of "active"
    # This creates a list of (timestamp, convo_id) for sorting
    all_convo_timestamps = []
    for cid, ts in project_convos.items():
        all_convo_timestamps.append((ts, cid, "project"))
    for cid, ts in message_convos.items():
        if cid not in project_convos: # Only add if not already covered by project timestamp
            all_convo_timestamps.append((ts, cid, "message"))
    
    # Sort by timestamp descending (most recent first)
    all_convo_timestamps.sort(key=lambda x: x[0], reverse=True)
    
    # Extract unique conversation IDs in sorted order
    sorted_unique_convos = []
    seen_convos = set()
    for _, cid, _ in all_convo_timestamps:
        if cid not in seen_convos:
            sorted_unique_convos.append(cid)
            seen_convos.add(cid)
            
    return {"conversations": sorted_unique_convos}


@app.get("/conversation/{conversation_id}")
def get_conversation_log_v1_5(conversation_id: str):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        SELECT role, content, agent_identifier, timestamp, raw_response
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id ASC
    """, (conversation_id,))
    rows = cur.fetchall()
    conn.close()
    messages = [{"role": r, "content": c, "agent_identifier": aid, "timestamp": ts, "raw_response": rr}
                for r, c, aid, ts, rr in rows]
    return {"messages": messages}

@app.post("/exec_command")
def exec_command_endpoint_v1_5(req: CommandExecRequest):
    if not req.command:
        raise HTTPException(status_code=400, detail="No command provided.")
    
    # If conversation_id is provided, command runs in that context.
    # Otherwise, it's a global command (runs in aggregator CWD, not advised for FS changes)
    convo_id_for_log = req.conversation_id or "GLOBAL_EXEC_COMMAND"
    
    stdout, stderr, code = run_command_and_log(req.command, convo_id_for_log)
    # Note: run_command_and_log already does DB logging.
    # Here, we just return the direct result of this specific call.
    return {"stdout": stdout, "stderr": stderr, "code": code}


@app.put("/project/{conversation_id}")
def put_project_v1_5(conversation_id: str, data: ProjectData):
    log_message_db(conversation_id, "system_project_control", "Received project data from frontend for saving/updating.", agent_identifier="FrontendSave")

    # ## START CHANGE ##
    # 1. Write the frontend's view of files to disk
    convo_folder = get_conversation_folder(conversation_id)
    
    # Clear existing files and folders in the workspace folder first (optional, but ensures fresh sync)
    # This is a strong operation, use with caution if you have other tools writing to this folder.
    # For a simple sync, it ensures the folder precisely matches the frontend's view.
    for item_name in os.listdir(convo_folder):
        item_path = os.path.join(convo_folder, item_name)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path) # Use shutil to remove directories
        except Exception as e:
            log_message_db(conversation_id, "system_error", f"Error clearing old workspace item {item_path}: {e}")

    # Write files from the frontend's workspaces data
    # Create a mapping for folders to handle hierarchy and parentId assignment more robustly
    # For initial implementation, we'll write all files and then rebuild the structure for the DB
    files_written_count = 0
    for ws_item in data.workspaces:
        if ws_item.get("type") == "folder":
            # For folders, just create the physical directory if it doesn't exist
            folder_path_on_disk = os.path.join(convo_folder, ws_item["id"]) # ID is relative path
            os.makedirs(folder_path_on_disk, exist_ok=True)
        else: # It's a file
            relative_file_path = ws_item.get("id") # Frontend's ID is the relative path
            if not relative_file_path: # Fallback if ID is not set as relative path yet
                relative_file_path = ws_item.get("title") # Use title as fallback if ID isn't relative path
            
            if not relative_file_path: continue # Skip if no title or ID

            # Sanitize path to prevent directory traversal
            sanitized_relative_path = os.path.normpath(relative_file_path).lstrip(os.sep)
            if sanitized_relative_path.startswith(".."):
                log_message_db(conversation_id, "system_security_warning", f"Blocked potentially malicious path during file write from frontend: {relative_file_path}")
                continue # Skip potentially malicious paths

            file_path_on_disk = os.path.join(convo_folder, sanitized_relative_path)
            
            try:
                os.makedirs(os.path.dirname(file_path_on_disk), exist_ok=True)
                with open(file_path_on_disk, "w", encoding="utf-8") as f:
                    f.write(ws_item.get("content", ""))
                files_written_count += 1
            except Exception as e:
                log_message_db(conversation_id, "system_error", f"Failed to write file '{sanitized_relative_path}' from frontend to disk: {e}")
                print(f"ERROR: Failed to write file {sanitized_relative_path} from frontend: {e}")
    log_message_db(conversation_id, "system_disk_write", f"Wrote {files_written_count} files from frontend data to disk for project {conversation_id}.")

    # 2. Update the project_json in the database with the received data (which now matches disk)
    # The frontend's 'data' object is the authoritative state for saving to DB in this flow
    project_dict_to_store = data.model_dump(exclude_none=True)
    _update_project_json_in_db(conversation_id, project_dict_to_store)
    # ## END CHANGE ##

    # Existing logging and checks remain
    if data.env:
        log_message_db(conversation_id, "system_project_control", f"Project has custom env config: {data.env}. Consider re-running dependency installs if applicable.")
    if os.path.exists(os.path.join(convo_folder, "package.json")):
        log_message_db(conversation_id, "system_project_control", "package.json found. If recently updated, 'npm install' might be needed.")
    if os.path.exists(os.path.join(convo_folder, "requirements.txt")):
        log_message_db(conversation_id, "system_project_control", "requirements.txt found. If recently updated, 'pip install -r requirements.txt' might be needed.")

    return {"status": "project_saved", "conversation_id": conversation_id}

@app.get("/project/{conversation_id}")
def get_project_v1_5(conversation_id: str):
    log_message_db(conversation_id, "system_project_control", "Attempting to load project data from DB and synchronize with disk.", agent_identifier="FrontendLoad")

    # 1. Load project JSON from the database (contains configuration and last known workspace state)
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT project_json, updated_at FROM projects WHERE conversation_id = ?", (conversation_id,))
    row = cur.fetchone()
    conn.close()

    db_project_data: Dict[str, Any] = {}
    updated_at = time.time() # Default for new project
    if row:
        project_json_str, updated_at = row
        try:
            db_project_data = json.loads(project_json_str)
        except json.JSONDecodeError:
            log_message_db(conversation_id, "system_error", "Failed to decode project JSON from DB, returning default structure.", agent_identifier="FrontendLoad")
            # Fall through to return default empty structure for this project
    else:
        log_message_db(conversation_id, "system_project_control", "No project data found in DB, starting new project.", agent_identifier="FrontendLoad")
    
    # ## START CHANGE ##
    # 2. Scan disk to get the authoritative file system state
    disk_workspaces = _scan_disk_for_workspaces(conversation_id)

    # 3. Merge: Prioritize disk files, but keep folders from DB if they are purely logical (not backed by a physical directory)
    # The `_scan_disk_for_workspaces` now returns a comprehensive list including folders,
    # so we can use its output as the primary source for the 'workspaces' list.
    # The goal is to return the actual disk state to the frontend.
    final_workspaces_for_frontend = disk_workspaces

    # Ensure empty logical folders from the DB (if any were created by frontend but no files written) are also included
    # This ensures consistency for purely UI-driven folder additions if they haven't been touched yet.
    # We primarily rely on the disk scan, but the DB could contain logical folders if they were added via UI only.
    db_logical_folders = {ws["id"] for ws in db_project_data.get("workspaces", []) if ws.get("type") == "folder"}
    disk_physical_folders = {ws["id"] for ws in disk_workspaces if ws.get("type") == "folder"}
    
    for folder_id in db_logical_folders:
        if folder_id not in disk_physical_folders:
            # If a folder was in DB but not physically on disk (e.g., empty folder created by UI only)
            # Add it to the final list, assuming `_scan_disk_for_workspaces` already created parentIds
            # This is a bit complex for a merge, often simpler to let _scan_disk handle all folder creation
            # and rely on it to ensure physical folders map to UI folders.
            # For this simplified sync, we assume _scan_disk_for_workspaces creates folder entries for all physically existing ones.
            # We don't merge "empty" DB folders unless they are explicitly created by the frontend as `type: "folder"` and saved.
            pass # No explicit merge of "empty" folders; `_scan_disk_for_workspaces` should be canonical

    # 4. Construct the full ProjectData object to return to the frontend
    # This ensures other project config (steps, initial prompt, etc.) are preserved
    # but the workspaces are strictly from the disk scan.
    full_data_to_return = ProjectData(
        initialPrompt=db_project_data.get("initialPrompt", ""),
        steps=db_project_data.get("steps", []),
        workspaces=final_workspaces_for_frontend, # THIS IS THE CRITICAL CHANGE
        customOrgs=db_project_data.get("customOrgs", []),
        contextAgentPort=db_project_data.get("contextAgentPort"),
        env=db_project_data.get("env"),
        pipelineSteps=db_project_data.get("pipelineSteps")
    ).model_dump()
    # ## END CHANGE ##

    log_message_db(conversation_id, "system_project_control", f"Project data loaded and synced with disk ({len(final_workspaces_for_frontend)} items).")
    return {"conversation_id": conversation_id, "updated_at": updated_at, **full_data_to_return}

# Memory, Vector, PR, Structured Data endpoints from V1, adapted for consistency
@app.post("/memory/put")
def put_memory_v1_5(req: MemoryRequest):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    now = time.time()
    cur.execute("""
        INSERT INTO memories (conversation_id, key, value, timestamp) VALUES (?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, timestamp=excluded.timestamp WHERE conversation_id=excluded.conversation_id
    """, (req.conversation_id, req.key, req.value, now))
    conn.commit()
    conn.close()
    log_message_db(req.conversation_id, "system_memory_control", f"Memory PUT: Key='{req.key}'")
    return {"status": "ok"}

@app.get("/memory/get")
def get_memory_v1_5(conversation_id: str, key: str):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT value FROM memories WHERE conversation_id = ? AND key = ? ORDER BY id DESC LIMIT 1", (conversation_id, key))
    row = cur.fetchone()
    conn.close()
    found = bool(row)
    value = row[0] if row else None
    log_message_db(conversation_id, "system_memory_control", f"Memory GET: Key='{key}', Found: {found}")
    return {"key": key, "value": value, "found": found}


@app.post("/vector/add")
def add_vector_v1_5(vec: VectorData):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    now = time.time()
    # Store embedding as JSON string as BLOB type can be tricky with SQLite and different drivers/languages
    emb_json = json.dumps(vec.embedding) 
    meta_json = json.dumps(vec.metadata) if isinstance(vec.metadata, dict) else str(vec.metadata)
    cur.execute("INSERT INTO vectors (conversation_id, embedding, metadata, timestamp) VALUES (?, ?, ?, ?)",
                (vec.conversation_id, emb_json, meta_json, now))
    new_id = cur.lastrowid
    conn.commit()
    conn.close()
    log_message_db(vec.conversation_id, "system_vector_db", f"Vector added, ID: {new_id}.")
    return {"status": "vector_added", "id": new_id}

@app.get("/vector/search")
def search_vector_v1_5(conversation_id: str, query_embedding_json: str, top_k: int = 3):
    try:
        q_emb = json.loads(query_embedding_json)
        if not isinstance(q_emb, list) or not all(isinstance(x, (int, float)) for x in q_emb):
            raise ValueError("Query embedding must be a list of numbers.")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid query_embedding_json: {e}")

    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    cur = conn.cursor()
    cur.execute("SELECT id, embedding, metadata, timestamp FROM vectors WHERE conversation_id = ?", (conversation_id,))
    rows = cur.fetchall()
    conn.close()

    results = []
    for row in rows:
        try:
            stored_emb = json.loads(row["embedding"])
            # Simple dot product for similarity score
            score = sum(qe * se for qe, se in zip(q_emb, stored_emb)) 
            results.append({
                "id": row["id"], "metadata": json.loads(row["metadata"]),
                "timestamp": row["timestamp"], "score": score
            })
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            log_message_db(conversation_id, "system_error", f"Error processing vector {row['id']} for search: {e}")
    results.sort(key=lambda x: x["score"], reverse=True) # Sort by score descending
    log_message_db(conversation_id, "system_vector_db", f"Vector search performed, found {len(results)} potential matches, returning top {top_k}.")
    return {"results": results[:top_k]}

def _approve_and_merge_pr_logic(conversation_id: str, pr_id: int, approver: str="SystemAuto") -> List[str]:
    """Shared logic for approving/merging PR. Returns list of changed files."""
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT pr_json, status FROM pull_requests WHERE id=? AND conversation_id=?", (pr_id, conversation_id))
    row = cur.fetchone()
    if not row:
        conn.close()
        log_message_db(conversation_id, "system_error", f"PR #{pr_id} not found for approval by {approver}.")
        raise ValueError(f"PR #{pr_id} not found.")
    
    pr_json_str, current_status = row
    if current_status == "merged":
        conn.close()
        log_message_db(conversation_id, "system_pr_control", f"PR #{pr_id} already merged. No action by {approver}.")
        return [] # Already merged

    pr_info = json.loads(pr_json_str)
    code_blocks_to_merge = pr_info.get("code_blocks", [])
    
    changed_files = _write_code_blocks_to_workspace(conversation_id, code_blocks_to_merge, f"PR_{pr_id}_{approver}")
    
    cur.execute("UPDATE pull_requests SET status='merged' WHERE id=? AND conversation_id=?", (pr_id, conversation_id))
    conn.commit()
    conn.close()
    log_message_db(conversation_id, "system_pr_control", f"PR #{pr_id} approved by {approver} and merged. Files affected: {changed_files}")
    
    if changed_files:
        auto_run_new_code_files(conversation_id, changed_files)
        # ## START CHANGE ##
        # After merging and writing files to disk, ensure DB project_json is updated
        _save_current_disk_state_to_db(conversation_id)
        # ## END CHANGE ##
    return changed_files

@app.post("/pr_submit")
def pr_submit_v1_5(data: PullRequestData):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    now = time.time()
    valid_blocks = [dict(cb) for cb in data.code_blocks if isinstance(cb, dict) and "title" in cb and "content" in cb]
    pr_info = {"code_blocks": valid_blocks, "description": data.description or ""}
    pr_json = json.dumps(pr_info)
    cur.execute("INSERT INTO pull_requests (conversation_id, pr_json, status, timestamp) VALUES (?, ?, ?, ?)",
                (data.conversation_id, pr_json, "pending", now))
    pr_id = cur.lastrowid
    conn.commit()
    conn.close()
    log_message_db(data.conversation_id, "system_pr_control", f"Pull Request #{pr_id} submitted: {data.description or 'No description'}")
    
    return {"status": "pr_submitted", "pr_id": pr_id, "conversation_id": data.conversation_id}

@app.get("/pr_list")
def pr_list_v1_5(conversation_id: str):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT id, pr_json, status, timestamp FROM pull_requests WHERE conversation_id = ? ORDER BY id ASC", (conversation_id,))
    rows = cur.fetchall()
    conn.close()
    results = []
    for row_data in rows:
        try:
            results.append({
                "id": row_data["id"], "pr_data": json.loads(row_data["pr_json"]),
                "status": row_data["status"], "timestamp": row_data["timestamp"]
            })
        except json.JSONDecodeError:
            results.append({"id": row_data["id"], "pr_data": {"error":"corrupted"}, "status": row_data["status"], "timestamp": row_data["timestamp"]})
    return {"pull_requests": results}

@app.post("/pr_approve")
def pr_approve_endpoint_v1_5(conversation_id: str, pr_id: int):
    try:
        changed_files = _approve_and_merge_pr_logic(conversation_id, pr_id, "UserManualApprove")
        return {"status": "pr_approved_and_merged", "pr_id": pr_id, "files_changed": changed_files}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error approving PR #{pr_id}: {str(e)}")

@app.post("/structured_data/add")
def add_structured_data_v1_5(data: StructuredData):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    now = time.time()
    json_data_str = json.dumps(data.json_data)
    cur.execute("INSERT INTO structured_data (conversation_id, data_type, json_data, timestamp) VALUES (?, ?, ?, ?)",
                (data.conversation_id, data.data_type, json_data_str, now))
    new_id = cur.lastrowid
    conn.commit()
    conn.close()
    log_message_db(data.conversation_id, "system_structured_data", f"Added structured data (Type: {data.data_type}, ID: {new_id})")
    return {"status": "ok", "id": new_id}

@app.get("/structured_data/list")
def list_structured_data_v1_5(conversation_id: str, data_type: Optional[str] = None):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if data_type:
        cur.execute("SELECT id, data_type, json_data, timestamp FROM structured_data WHERE conversation_id=? AND data_type=? ORDER BY id ASC", (conversation_id, data_type))
    else:
        cur.execute("SELECT id, data_type, json_data, timestamp FROM structured_data WHERE conversation_id=? ORDER BY id ASC", (conversation_id,))
    rows = cur.fetchall()
    conn.close()
    results = []
    for row_data in rows:
        try:
            results.append({
                "id": row_data["id"], "data_type": row_data["data_type"],
                "json_data": json.loads(row_data["json_data"]), "timestamp": row_data["timestamp"]
            })
        except json.JSONDecodeError:
            results.append({"id": row_data["id"], "data_type": row_data["data_type"], "json_data": {"error":"corrupted"}, "timestamp": row_data["timestamp"]})
    return {"structured_data": results}

@app.post("/context_agent_call") # Logging conceptual call
def context_agent_call_log_v1_5(req: LogRequest):
    log_identifier = req.agent_identifier_obj.name if req.agent_identifier_obj and req.agent_identifier_obj.name \
                     else str(req.agent_port or "ContextAgentLog")
    log_message_db(req.conversation_id, req.role, req.content, log_identifier)
    return {"status": "context_agent_interaction_logged"}

# --- Auto Pipeline (Enhanced from V2) ---
def fetch_custom_pipeline_steps(conversation_id: str) -> Optional[List[Dict[str, str]]]:
    try:
        # Use get_project_v1_5 to retrieve project data with pydantic validation
        project_data_full = get_project_v1_5(conversation_id) 
        pipeline_steps_cfg = project_data_full.get("pipelineSteps")
        if pipeline_steps_cfg and isinstance(pipeline_steps_cfg, list):
            # Validate structure of pipeline steps to ensure they have name and cmd
            if all(isinstance(s, dict) and "name" in s and "cmd" in s for s in pipeline_steps_cfg):
                return pipeline_steps_cfg
            else: # Log if structure is invalid, but still exists
                log_message_db(conversation_id, "system_warning", "Custom pipelineSteps found but malformed in project_json.")
        return None
    except HTTPException as he: # Project not found (404) is acceptable, others should still raise
        if he.status_code == 404: return None
        raise
    except Exception as e:
        log_message_db(conversation_id, "system_error", f"Error fetching/parsing custom pipeline steps: {e}")
        return None

def _run_pipeline_step_v1_5(conversation_id: str, step_name: str, command: str, timeout: int=300) -> Dict[str, Any]:
    log_message_db(conversation_id, "pipeline_step_start", f"Running pipeline step: '{step_name}', Command: '{command}'", agent_identifier="PipelineRunner")
    stdout, stderr, rc = run_command_and_log(command, conversation_id, timeout_seconds=timeout)
    status = "passed" if rc == 0 else "failed"
    detail_log = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
    _insert_pipeline_run_db_v1_5(conversation_id, step_name, status, detail_log)
    result = {"step_name": step_name, "status": status, "detail": detail_log, "return_code": rc}
    if status == "failed":
        repeated_msg = _check_repeated_pipeline_fails_db_v1_5(conversation_id, step_name)
        if repeated_msg: result["supervisor_suggestion"] = repeated_msg
    return result

def _insert_pipeline_run_db_v1_5(conversation_id: str, step_name: str, status: str, detail: Optional[str]=None):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("INSERT INTO pipeline_runs (conversation_id, step_name, status, detail, timestamp) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, step_name, status, detail, time.time()))
    conn.commit()
    conn.close()

def _check_repeated_pipeline_fails_db_v1_5(conversation_id: str, step_name: str, threshold: int=2) -> Optional[str]:
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT status FROM pipeline_runs WHERE conversation_id=? AND step_name=? ORDER BY id DESC LIMIT ?",
                (conversation_id, step_name, threshold))
    rows = cur.fetchall()
    conn.close()
    if len(rows) < threshold: return None
    fail_count = sum(1 for r_status in rows if r_status[0] == 'failed')
    if fail_count >= threshold:
        return f"Pipeline Supervisor: Step '{step_name}' has failed {threshold}+ times consecutively/recently. Review needed."
    return None

@app.post("/auto_pipeline")
def auto_pipeline_v1_5(req: PipelineRequest):
    conversation_id = req.conversation_id
    log_message_db(conversation_id, "pipeline_control", "Auto-pipeline initiated.", agent_identifier="PipelineTrigger")
    
    steps_to_run_cfg = fetch_custom_pipeline_steps(conversation_id)
    if not steps_to_run_cfg: # Fallback to request defaults if no custom steps in project
        log_message_db(conversation_id, "pipeline_control", "Using default build, test, lint from request (no project config).", agent_identifier="PipelineConfig")
        steps_to_run_cfg = [
            {"name": "build", "cmd": req.build_command or "echo 'Build step skipped (no command).'"},
            {"name": "test", "cmd": req.test_command or "echo 'Test step skipped (no command).'"},
            {"name": "lint", "cmd": req.lint_command or "echo 'Lint step skipped (no command).'"}
        ]
    
    all_passed = True
    run_details_list = []
    for step_cfg in steps_to_run_cfg:
        step_name_cfg = step_cfg.get("name", "unnamed_step")
        cmd_cfg = step_cfg.get("cmd")
        timeout_cfg = int(step_cfg.get("timeout", 300)) # Default 5 min
        if not cmd_cfg:
            log_message_db(conversation_id, "pipeline_warning", f"Skipping step '{step_name_cfg}': no command configured.", agent_identifier="PipelineRunner")
            run_details_list.append({"step_name": step_name_cfg, "status": "skipped", "detail": "No command configured."})
            continue
        
        step_run_result = _run_pipeline_step_v1_5(conversation_id, step_name_cfg, cmd_cfg, timeout_cfg)
        run_details_list.append(step_run_result)
        if step_run_result["status"] == "failed":
            all_passed = False
            log_message_db(conversation_id, "pipeline_control", f"Pipeline halted at failed step: {step_name_cfg}.", agent_identifier="PipelineRunner")
            break
            
    final_pipeline_status = "success" if all_passed else "failure"
    merged_pr_ids_list = []
    if all_passed:
        log_message_db(conversation_id, "pipeline_control", "All pipeline steps passed. Merging pending PRs.", agent_identifier="PipelineRunner")
        conn_pr_lookup = sqlite3.connect(DB_NAME)
        cur_pr_lookup = conn_pr_lookup.cursor()
        cur_pr_lookup.execute("SELECT id FROM pull_requests WHERE conversation_id=? AND status='pending'", (conversation_id,))
        pending_ids = [r[0] for r in cur_pr_lookup.fetchall()]
        conn_pr_lookup.close()
        for pr_id_merge in pending_ids:
            try:
                _approve_and_merge_pr_logic(conversation_id, pr_id_merge, "PipelineAutoApprove")
                merged_pr_ids_list.append(pr_id_merge)
            except Exception as e_merge:
                log_message_db(conversation_id, "pipeline_error", f"Failed to auto-merge PR #{pr_id_merge}: {e_merge}", agent_identifier="PipelineRunner")
    
    response_msg = f"Pipeline {'completed successfully' if all_passed else 'failed'}. Merged {len(merged_pr_ids_list)} PR(s)."
    failing_step = next((s for s in run_details_list if s["status"] == "failed"), None)

    return {
        "status": final_pipeline_status,
        "message": response_msg,
        "merged_prs": merged_pr_ids_list,
        "pipeline_run_details": run_details_list,
        "supervisor_suggestion": failing_step.get("supervisor_suggestion") if failing_step else None
    }

# --- Auto Code Runner (Enhanced from V2) ---
def auto_run_new_code_files(conversation_id: str, changed_filenames: Optional[List[str]] = None, max_runs: int = 1):
    """Runs .py, .js, .sh files if new or changed, up to max_runs."""
    convo_folder = get_conversation_folder(conversation_id)
    files_to_scan = []
    if changed_filenames:
        # Filter for files that are executable scripts
        files_to_scan = [fn for fn in changed_filenames if fn.lower().endswith((".py", ".js", ".sh"))]
    else: # Scan entire folder if no specific changed files are provided
        try:
            # os.listdir only returns immediate contents, os.walk is better for recursive scan
            for root, _, files in os.walk(convo_folder):
                for file_name in files:
                    if file_name.lower().endswith((".py", ".js", ".sh")):
                        # Add relative path to the list
                        files_to_scan.append(os.path.relpath(os.path.join(root, file_name), convo_folder))
        except FileNotFoundError:
            log_message_db(conversation_id, "system_warning", f"Workspace folder not found for auto-run scan: {convo_folder}")
            return
    
    if not files_to_scan: return

    log_message_db(conversation_id, "system_auto_code_run", f"Checking files for auto-run: {files_to_scan}", agent_identifier="CodeRunner")
    for fname in files_to_scan:
        _maybe_run_code_file_v1_5(conversation_id, fname, max_runs)

def _maybe_run_code_file_v1_5(conversation_id: str, filename: str, max_runs_per_file: int):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    # Check if this file has been run before in this conversation context
    cur.execute("SELECT id, run_count FROM executed_code_files WHERE conversation_id=? AND filename=?", (conversation_id, filename))
    row = cur.fetchone()
    db_id, current_run_count = (row[0], row[1]) if row else (None, 0)

    if current_run_count >= max_runs_per_file:
        log_message_db(conversation_id, "system_auto_code_run", f"Skipping auto-run for '{filename}', max runs ({max_runs_per_file}) met.", agent_identifier="CodeRunner")
        conn.close()
        return

    cmd_prefix = None
    if filename.lower().endswith(".py"): cmd_prefix = "python"
    elif filename.lower().endswith(".js"): cmd_prefix = "node"
    elif filename.lower().endswith(".sh"): cmd_prefix = "bash" # or sh
    if not cmd_prefix:
        conn.close()
        return

    # Command to execute is relative to the conversation folder, so `run_command_and_log`
    # will handle changing directory correctly.
    command_to_execute = f"{cmd_prefix} \"{filename}\"" 
    _, _, rc_code = run_command_and_log(command_to_execute, conversation_id)
    new_count = current_run_count + 1
    
    # Update or insert the record of this file's execution
    if db_id is None:
        cur.execute("INSERT INTO executed_code_files (conversation_id, filename, last_run_time, last_return_code, run_count) VALUES (?, ?, ?, ?, ?)",
                    (conversation_id, filename, time.time(), rc_code, new_count))
    else:
        cur.execute("UPDATE executed_code_files SET last_run_time=?, last_return_code=?, run_count=? WHERE id=?",
                    (time.time(), rc_code, new_count, db_id))
    conn.commit()
    conn.close()
    log_message_db(conversation_id, "system_auto_code_run", f"Auto-run for '{filename}' (attempt #{new_count}) completed, RC: {rc_code}.", agent_identifier="CodeRunner")


# --- Background Threads & Lifecycle ---
STOP_BG_THREADS_EVENT = threading.Event()
LAST_MESSAGE_ID_CHECKED_FOR_BG_CMDS = 0

def background_command_detector_v1_5():
    global LAST_MESSAGE_ID_CHECKED_FOR_BG_CMDS
    log_message_db("GLOBAL_SYSTEM_LOG", "system_bg_detector", "Background command/file detector thread starting.", agent_identifier="System")
    try: # Initialize last checked ID by getting the max ID from messages
        conn_init_bg = sqlite3.connect(DB_NAME)
        cur_init_bg = conn_init_bg.cursor()
        cur_init_bg.execute("SELECT MAX(id) FROM messages")
        max_id_row_bg = cur_init_bg.fetchone()
        if max_id_row_bg and max_id_row_bg[0] is not None:
            LAST_MESSAGE_ID_CHECKED_FOR_BG_CMDS = max_id_row_bg[0]
        conn_init_bg.close()
        log_message_db("GLOBAL_SYSTEM_LOG", "system_bg_detector", f"Initialized BG command detector last_id to: {LAST_MESSAGE_ID_CHECKED_FOR_BG_CMDS}", agent_identifier="System")
    except Exception as e_init_bg_cmd:
         log_message_db("GLOBAL_SYSTEM_LOG", "system_error", f"Error initializing BG command detector last_id: {e_init_bg_cmd}", agent_identifier="System")

    while not STOP_BG_THREADS_EVENT.is_set():
        try:
            conn_bg = sqlite3.connect(DB_NAME)
            cur_bg = conn_bg.cursor()
            # Process new messages from agent-like roles or system outputs that might contain directives
            cur_bg.execute("""
                SELECT id, conversation_id, role, content FROM messages
                WHERE id > ? AND (
                    role LIKE 'agent%' OR               -- Catch all agent responses (e.g., agent_response_from_Coder)
                    role LIKE 'user_to_interactive_agent' OR -- User input to interactive step
                    role LIKE '%_interactive_agent' OR  -- Responses from interactive agents (like Coder)
                    role LIKE 'system_chain_final_output' OR -- Final output from an auto-chain
                    role LIKE 'chain_step_%'            -- Responses within chain steps
                )
                ORDER BY id ASC LIMIT 100 
            """, (LAST_MESSAGE_ID_CHECKED_FOR_BG_CMDS,)) # Process in batches
            new_messages_bg = cur_bg.fetchall()
            conn_bg.close()

            if new_messages_bg:
                max_proc_id_batch = LAST_MESSAGE_ID_CHECKED_FOR_BG_CMDS
                convo_ids_with_changes = set() # Track unique convo IDs that had disk changes
                for msg_id, convo_id, msg_role, msg_content in new_messages_bg:
                    if msg_content: # Ensure content is not None
                        run_terminal_commands_from_text(msg_content, convo_id, msg_role) # Checks for TERMINAL_CMD and shell blocks
                        
                        # advanced_parse_and_handle_file_directives itself calls _write_code_blocks_to_workspace
                        # which now calls _save_current_disk_state_to_db internally.
                        # So, we just need to ensure this is called.
                        changed_by_file_directives = advanced_parse_and_handle_file_directives(msg_content, convo_id, msg_role) 
                        
                        if changed_by_file_directives:
                            auto_run_new_code_files(convo_id, changed_by_file_directives)
                            # The _save_current_disk_state_to_db is now handled within _write_code_blocks_to_workspace.
                            # So, no need to call it here directly.
                            # Just mark that this convo had changes.
                            convo_ids_with_changes.add(convo_id)
                    max_proc_id_batch = max(max_proc_id_batch, msg_id)
                LAST_MESSAGE_ID_CHECKED_FOR_BG_CMDS = max_proc_id_batch
                
                # ## START CHANGE ##
                # (Removed direct _save_current_disk_state_to_db here, as it's now nested deeper in _write_code_blocks_to_workspace)
                # ## END CHANGE ##

            else: # No new messages, wait a bit longer to reduce CPU cycles
                STOP_BG_THREADS_EVENT.wait(5) # Wait 5s if no messages
                continue
        except sqlite3.OperationalError as oe_bg:
            # This can happen if another thread is writing to the DB or DB is locked.
            # Log and wait, then retry.
            log_message_db("GLOBAL_SYSTEM_LOG", "system_error", f"BG command detector DB error (will retry): {oe_bg}", agent_identifier="System")
            STOP_BG_THREADS_EVENT.wait(10)
            continue
        except Exception as e_bg_cmd:
            log_message_db("GLOBAL_SYSTEM_LOG", "system_error", f"Background command/file detector error: {e_bg_cmd}", agent_identifier="System")
        
        STOP_BG_THREADS_EVENT.wait(2) # Poll frequency if there were messages
    log_message_db("GLOBAL_SYSTEM_LOG", "system_bg_detector", "Background command/file detector thread stopped.", agent_identifier="System")


# --- Helper for file type determination (Backend version, matching frontend logic) ---
def determineTypeFromFilename(filename: str, lang_hint: Optional[str] = None) -> str:
    """Determine a generic type for a workspace item based on filename or language hint."""
    name_lower = filename.lower()
    lang_lower = (lang_hint or "").lower()

    # Priority 1: Specific extensions/languages
    if name_lower.endswith((".py", ".pyw")) or lang_lower == "python": return "code"
    if name_lower.endswith((".js", ".jsx", ".ts", ".tsx")) or lang_lower in ["javascript", "typescript", "jsx", "tsx"]: return "code"
    if name_lower.endswith((".html", ".htm")) or lang_lower in ["html", "htmlmixed"]: return "html"
    if name_lower.endswith(".css") or lang_lower == "css": return "css"
    if name_lower.endswith(".sh") or lang_lower in ["shell", "bash", "sh"]: return "code" # shell scripts are code
    if name_lower.endswith((".md", ".markdown")): return "text" # Markdown treated as text
    if name_lower.endswith(".json") or lang_lower == "json": return "text" # JSON usually edited as text
    if name_lower.endswith(".xml") or lang_lower == "xml": return "text" # XML usually edited as text

    # Priority 2: Generic language hint (if not "text" or "plaintext")
    if lang_lower and lang_lower not in ["text", "plaintext"]: return "code"

    return "text" # Default for unknown types



# --- NEW: Core Disk Scanning Function ---
def _scan_disk_for_workspaces(conversation_id: str) -> List[Dict[str, Any]]:
    """
    Scans the conversation's workspace folder on disk and returns a list of
    workspace items (files and folders) compatible with the frontend's structure.
    Generates stable IDs based on relative paths.
    """
    convo_folder = get_conversation_folder(conversation_id)
    workspaces_from_disk: List[Dict[str, Any]] = []
    
    # Store already generated folder IDs to avoid duplicates and ensure parentId matches
    # Maps relative_path -> generated_id (which is also the relative_path)
    generated_folder_ids = set()

    # Add root folder explicitly if it doesn't exist logically
    # This ensures the base folder always appears in the UI
    if not os.path.exists(convo_folder):
        os.makedirs(convo_folder, exist_ok=True)
    
    # Walk directory in a sorted manner for consistent ordering
    for root, dirs, files in os.walk(convo_folder):
        # Sort dirs and files for consistent UI order
        dirs.sort()
        files.sort()

        relative_root = os.path.relpath(root, convo_folder)
        if relative_root == ".":
            relative_root = "" # Root of the project

        current_folder_id = relative_root if relative_root else None # Root folder has no parentId
        
        # Add folders (if not already added as a parent of a file)
        for d in dirs:
            relative_folder_path = os.path.join(relative_root, d).replace("\\", "/") # Standardize path separators
            if relative_folder_path not in generated_folder_ids:
                workspaces_from_disk.append({
                    "id": relative_folder_path,
                    "type": "folder",
                    "title": d,
                    "parentId": current_folder_id,
                    "content": "", # Folders have no content
                    "lang": "", # Folders have no lang
                    "versionHistory": [] # No version history for folders from disk scan
                })
                generated_folder_ids.add(relative_folder_path)

        # Add files
        for f in files:
            file_path_full = os.path.join(root, f)
            relative_file_path = os.path.relpath(file_path_full, convo_folder).replace("\\", "/") # Standardize path separators
            
            # Skip hidden files or system files if desired (e.g., .DS_Store, Thumbs.db)
            if f.startswith(".") or f == "Thumbs.db":
                continue

            try:
                with open(file_path_full, 'r', encoding='utf-8') as file_obj:
                    file_content = file_obj.read()
            except UnicodeDecodeError:
                file_content = f"[Binary file: {f} - Content not displayed]"
                log_message_db(conversation_id, "system_warning", f"Skipping binary file '{relative_file_path}' in disk scan (cannot decode).")
            except Exception as e:
                file_content = f"[Error reading file: {e}]"
                log_message_db(conversation_id, "system_error", f"Error reading file '{relative_file_path}' during disk scan: {e}")

            # Determine type using the backend helper
            file_type = determineTypeFromFilename(f, os.path.splitext(f)[1].lstrip('.'))
            file_lang = os.path.splitext(f)[1].lstrip('.') # Simple extension as lang hint

            workspaces_from_disk.append({
                "id": relative_file_path,
                "type": file_type,
                "title": f,
                "content": file_content,
                "lang": file_lang,
                "parentId": current_folder_id,
                "versionHistory": [{"content": file_content, "timestamp": time.time()}] # Initialize with current disk content
            })
    
    # Ensure consistent sorting for predictable UI
    workspaces_from_disk.sort(key=lambda x: (x.get("type", "z"), x.get("title", "")))

    return workspaces_from_disk



def _update_project_json_in_db(conversation_id: str, project_data_dict: Dict[str, Any]):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    now = time.time()
    
    serialized = json.dumps(project_data_dict, ensure_ascii=False)
    cur.execute("""
        INSERT INTO projects (conversation_id, project_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(conversation_id) DO UPDATE SET
            project_json = excluded.project_json,
            updated_at = excluded.updated_at
    """, (conversation_id, serialized, now))
    conn.commit()
    conn.close()
    log_message_db(conversation_id, "system_project_control", "Project data (JSON) saved/updated in DB.")
    
    

# --- NEW: Helper to synchronize disk state back to DB's project_json ---
def _save_current_disk_state_to_db(conversation_id: str):
    log_message_db(conversation_id, "system_sync_disk_to_db", "Initiating disk scan to update DB project_json.", agent_identifier="SyncManager")
    try:
        # 1. Get current project data from DB
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT project_json FROM projects WHERE conversation_id = ?", (conversation_id,))
        row = cur.fetchone()
        conn.close()

        current_project_data = {}
        if row and row[0]:
            try:
                current_project_data = json.loads(row[0])
            except json.JSONDecodeError:
                log_message_db(conversation_id, "system_error", "Failed to decode existing project_json from DB during sync, starting fresh for workspaces.", agent_identifier="SyncManager")
        
        # 2. Scan disk for current workspace state
        disk_workspaces = _scan_disk_for_workspaces(conversation_id)
        
        # 3. Update the 'workspaces' field in the project_data
        current_project_data["workspaces"] = disk_workspaces

        # 4. Save the updated project data back to DB
        _update_project_json_in_db(conversation_id, current_project_data)
        log_message_db(conversation_id, "system_sync_disk_to_db", "Successfully updated DB project_json from disk scan.", agent_identifier="SyncManager")

    except Exception as e:
        log_message_db(conversation_id, "system_error", f"Error during disk-to-DB sync for project {conversation_id}: {e}", agent_identifier="SyncManager")
        print(f"ERROR: Disk-to-DB sync failed for {conversation_id}: {e}")


def housekeeping_thread_v1_5():
    log_message_db("GLOBAL_SYSTEM_LOG", "system_housekeeping", "Housekeeping thread starting.", agent_identifier="System")
    while not STOP_BG_THREADS_EVENT.is_set():
        try:
            # Example: Delete messages older than 60 days
            # sixty_days_ago = time.time() - (60 * 24 * 60 * 60)
            # conn_hk = sqlite3.connect(DB_NAME)
            # cur_hk = conn_hk.cursor()
            # cur_hk.execute("DELETE FROM messages WHERE timestamp < ?", (sixty_days_ago,))
            # deleted_count_hk = cur_hk.rowcount
            # conn_hk.commit()
            # conn_hk.close()
            # if deleted_count_hk > 0:
            #     log_message_db("GLOBAL_SYSTEM_LOG", "system_housekeeping", f"Cleaned up {deleted_count_hk} old messages.", agent_identifier="System")
            pass # Add more housekeeping tasks if needed
        except Exception as e_hk_thread:
            log_message_db("GLOBAL_SYSTEM_LOG", "system_error", f"Housekeeping error: {e_hk_thread}", agent_identifier="System")
        
        STOP_BG_THREADS_EVENT.wait(3600) # Run hourly
    log_message_db("GLOBAL_SYSTEM_LOG", "system_housekeeping", "Housekeeping thread stopped.", agent_identifier="System")

# FastAPI Lifecycle Events
async def startup_event():
    log_message_db("GLOBAL_SYSTEM_LOG", "system_lifecycle", f"Aggregator V1.5.1 Startup: PID={os.getpid()}", agent_identifier="System")
    # Fetch agents once on startup to populate the global cache for summarization
    # We call the function directly to ensure the cache is warmed up
    # before any requests come in.
    get_agents_v1_5() 
    # Start background threads
    app.state.bg_command_detector_thread = threading.Thread(target=background_command_detector_v1_5, daemon=True)
    app.state.bg_command_detector_thread.start()
    app.state.housekeeping_thread = threading.Thread(target=housekeeping_thread_v1_5, daemon=True)
    app.state.housekeeping_thread.start()
@app.get("/")
async def get_index_html():
    # Assumes index.html is in the same directory as aggregator.py
    # (which will be /app/ in the Docker container)
    return FileResponse("index.html")


async def shutdown_event():
    log_message_db("GLOBAL_SYSTEM_LOG", "system_lifecycle", "Aggregator V1.5.1 Shutdown sequence initiated.", agent_identifier="System")
    STOP_BG_THREADS_EVENT.set() # Signal background threads to stop
    # Give threads a chance to finish cleanly
    if hasattr(app.state, 'bg_command_detector_thread') and app.state.bg_command_detector_thread.is_alive():
        app.state.bg_command_detector_thread.join(timeout=10)
    if hasattr(app.state, 'housekeeping_thread') and app.state.housekeeping_thread.is_alive():
        app.state.housekeeping_thread.join(timeout=10)
    log_message_db("GLOBAL_SYSTEM_LOG", "system_lifecycle", "Aggregator V1.5.1 Shutdown complete.", agent_identifier="System")

app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

if __name__ == "__main__":
    print("Starting Aggregator V1.5.1 Server...")
    # init_db() is called at module level already
    # Uvicorn will call startup_event and shutdown_event automatically
    uvicorn.run(app, host="0.0.0.0", port=8000)
