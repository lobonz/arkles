from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import requests
import json
import re
import os
import time
import uuid
import traceback
import sqlite3
from datetime import datetime
from typing import Dict, List, Any

app = Flask(__name__)
CORS(app)

class OfficeRequirementsAgent:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "qwen3"  # Adjust model name as needed
        self.debug = True #os.getenv("LLM_DEBUG", "1") != "0"
        # Provider config: set to 'ollama' or 'openrouter'
        self.provider = "ollama"
        # OpenRouter config (edit directly; no env vars)
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.openrouter_api_key = "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"
        # OpenRouter models list (can be overridden per request). Adjust slugs as needed.
        self.openrouter_models: List[str] = [
            "moonshotai/kimi-k2:free",
            "openai/gpt-oss-20b:free",
            "mistralai/mistral-small-3.2-24b-instruct:free"
        ]
        # Back-compat default
        self.openrouter_model = self.openrouter_models[0]
        # Optional attribution headers recommended by OpenRouter
        self.openrouter_referer = ""  # e.g., https://your-site.example
        self.openrouter_title = ""    # e.g., Your App Name
        # Simple SQLite store (for testing)
        self.db_path = "office_assistant.db"
        self.init_database()
        # Interview behavior controls
        self.max_followup_questions_per_turn = 2
        self.max_assistant_turns = 8  # safety cap only; not used for auto-close
        self.enable_auto_close = True
        self.closing_message = (
            "Thanks for all the details — we'll get to work finding the right space for you and circle back with options shortly."
        )
        # Persona and priorities
        self.persona = (
            "You are a professional real estate broker who is excellent at quickly understanding client needs. "
            "Be concise, to the point, and avoid repeating questions already answered."
        )
        # Essential fields required to consider the interview complete
        self.essential_fields = [
            "employees", "business_type", "budget_range", "timeline", "preferred_areas",
            "working_style", "layout_preference", "meeting_rooms", "required_size"
        ]

        # Interview mode: 'core' or 'full'
        self.interview_mode = "full"
        # Completion mode controls when to close: 'essentials' or 'full'
        self.completion_mode = "full"
        # Sections mapped to fields for broader checklist coverage
        self.sections = {
            "Business Profile": ["employees", "business_type"],
            "Existing Space": ["has_existing_space", "current_location", "current_cost", "required_size"],
            "Strategic Location": ["preferred_areas", "proximity_requirements", "transport_access", "brand_image"],
            "Size & Layout": ["working_style", "layout_preference", "meeting_rooms"],
            "Cost & Lease": ["budget_range", "lease_term", "incentives_needed", "fitout_budget"],
            "Building & Amenities": ["building_services", "shared_amenities", "security_needs"],
            "Technology": ["internet_requirements", "power_requirements", "data_infrastructure"],
            "Legal & Compliance": ["compliance_needs"],
            "Employee Considerations": ["natural_light"],
            "Sustainability": ["sustainability_importance", "green_credentials"],
            "Flexibility": ["flexibility_needs", "short_term_options"],
        }

        # Provider config: set to 'ollama' or 'openrouter'
        self.provider = "openrouter"
        # OpenRouter config (edit directly; no env vars)
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.openrouter_api_key = "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"
        # OpenRouter models list (can be overridden per request). Adjust slugs as needed.
        self.openrouter_models: List[str] = [
            "moonshotai/kimi-k2:free",
            "google/gemini-1.5-flash-latest",
            "mistralai/mistral-nemo:free"
        ]
        # Back-compat default
        self.openrouter_model = self.openrouter_models[0]
        # Optional attribution headers recommended by OpenRouter
        self.openrouter_referer = ""  # e.g., https://your-site.example
        self.openrouter_title = ""    # e.g., Your App Name

        # Test connection and get available models on startup
        if self.provider == "ollama":
            self.verify_ollama_connection()
        else:
            self._debug_print("[LLM] Using OpenRouter provider. Skipping local Ollama model discovery.")

        # Define the comprehensive requirements schema
        self.requirements_schema = {
            "employees": "Number of employees (integer)",
            "business_type": "Type of business (e.g., tech startup, accounting firm, consulting)",
            "has_existing_space": "Whether they currently have office space (yes/no)",
            "current_location": "Current office location if applicable",
            "current_cost": "Current annual cost if applicable",
            "required_size": "Required office size in square meters",
            "working_style": "Working style (office-based, hybrid, flexible)",
            "layout_preference": "Layout preference (open plan, mixed, closed offices)",
            "meeting_rooms": "Meeting room requirements",
            "growth_plans": "Plans for business growth",
            "preferred_areas": "Preferred locations/suburbs",
            "transport_access": "Transport and accessibility needs",
            "brand_image": "Importance of location for brand image",
            "proximity_requirements": "Need to be close to clients/suppliers/etc",
            "budget_range": "Annual budget range",
            "lease_term": "Preferred lease term length",
            "incentives_needed": "Required lease incentives",
            "fitout_budget": "Available fit-out budget",
            "internet_requirements": "Internet and connectivity needs",
            "power_requirements": "Power and electrical needs",
            "data_infrastructure": "Data and IT infrastructure needs",
            "building_services": "Required building services (lifts, HVAC, etc)",
            "shared_amenities": "Desired shared amenities",
            "security_needs": "Security requirements",
            "natural_light": "Importance of natural light and airflow",
            "sustainability_importance": "Importance of sustainability/green features",
            "green_credentials": "Specific green building requirements",
            "compliance_needs": "Compliance and regulatory requirements",
            "flexibility_needs": "Lease flexibility requirements",
            "timeline": "Required move-in timeline",
            "short_term_options": "Interest in short-term/flexible workspace"
        }

        # After base config, run provider checks
        if self.provider == "ollama":
            self.verify_ollama_connection()
        else:
            self._debug_print("[LLM] Using OpenRouter provider. Skipping local Ollama model discovery.")

        # Define the comprehensive requirements schema
        self.requirements_schema = {
            "employees": "Number of employees (integer)",
            "business_type": "Type of business (e.g., tech startup, accounting firm, consulting)",
            "has_existing_space": "Whether they currently have office space (yes/no)",
            "current_location": "Current office location if applicable",
            "current_cost": "Current annual cost if applicable",
            "required_size": "Required office size in square meters",
            "working_style": "Working style (office-based, hybrid, flexible)",
            "layout_preference": "Layout preference (open plan, mixed, closed offices)",
            "meeting_rooms": "Meeting room requirements",
            "growth_plans": "Plans for business growth",
            "preferred_areas": "Preferred locations/suburbs",
            "transport_access": "Transport and accessibility needs",
            "brand_image": "Importance of location for brand image",
            "proximity_requirements": "Need to be close to clients/suppliers/etc",
            "budget_range": "Annual budget range",
            "lease_term": "Preferred lease term length",
            "incentives_needed": "Required lease incentives",
            "fitout_budget": "Available fit-out budget",
            "internet_requirements": "Internet and connectivity needs",
            "power_requirements": "Power and electrical needs",
            "data_infrastructure": "Data and IT infrastructure needs",
            "building_services": "Required building services (lifts, HVAC, etc)",
            "shared_amenities": "Desired shared amenities",
            "security_needs": "Security requirements",
            "natural_light": "Importance of natural light and airflow",
            "sustainability_importance": "Importance of sustainability/green features",
            "green_credentials": "Specific green building requirements",
            "compliance_needs": "Compliance and regulatory requirements",
            "flexibility_needs": "Lease flexibility requirements",
            "timeline": "Required move-in timeline",
            "short_term_options": "Interest in short-term/flexible workspace"
        }

        # ---------- Field planning helpers ----------
    def get_all_section_fields(self) -> List[str]:
        all_fields: List[str] = []
        for _, fields in self.sections.items():
            for f in fields:
                if f not in all_fields:
                    all_fields.append(f)
        return all_fields

    def get_completion_fields(self, completion_mode: str | None) -> List[str]:
        mode = (completion_mode or self.completion_mode or "essentials").lower()
        if mode == "full":
            return self.get_all_section_fields()
        return self.essential_fields

    def get_prioritized_fields(self, current_requirements: Dict[str, Any], interview_mode: str | None) -> List[str]:
        mode = (interview_mode or self.interview_mode or "core").lower()
        if mode == "core":
            return [k for k in self.essential_fields if not current_requirements.get(k)]
        # 'full': walk sections
        prioritized: List[str] = []
        for _, fields in self.sections.items():
            for f in fields:
                if f not in prioritized and not current_requirements.get(f):
                    prioritized.append(f)
        return prioritized

    # ---------- SQLite helpers ----------
    def init_database(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        provider TEXT,
                        model TEXT
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS requirements_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL,
                        requirements_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_req_conv ON requirements_snapshots(conversation_id, id)")
                conn.commit()
        except Exception:
            self._debug_print("[DB] Failed to initialize database")
            self._debug_print(traceback.format_exc())

    @staticmethod
    def _utc_now() -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def create_conversation(self, provider: str | None, model: str | None) -> str:
        conversation_id = str(uuid.uuid4())
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO conversations (id, created_at, provider, model) VALUES (?, ?, ?, ?)",
                    (
                        conversation_id,
                        self._utc_now(),
                        (provider or self.provider),
                        (model or (self.openrouter_model if (provider or self.provider) == "openrouter" else self.model)),
                    ),
                )
                conn.commit()
        except Exception:
            self._debug_print("[DB] Failed to create conversation")
            self._debug_print(traceback.format_exc())
        return conversation_id

    def save_message(self, conversation_id: str, role: str, content: str) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (conversation_id, role, content, self._utc_now()),
                )
                conn.commit()
        except Exception:
            self._debug_print(f"[DB] Failed to save message for conversation {conversation_id}")
            self._debug_print(traceback.format_exc())

    def save_requirements_snapshot(self, conversation_id: str, requirements: Dict[str, Any]) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO requirements_snapshots (conversation_id, requirements_json, created_at) VALUES (?, ?, ?)",
                    (conversation_id, json.dumps(requirements), self._utc_now()),
                )
                conn.commit()
        except Exception:
            self._debug_print(f"[DB] Failed to save requirements for conversation {conversation_id}")
            self._debug_print(traceback.format_exc())

    def load_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute(
                    "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                    (conversation_id,),
                )
                rows = cur.fetchall()
                return [{"role": row["role"], "content": row["content"]} for row in rows]
        except Exception:
            self._debug_print(f"[DB] Failed to load conversation {conversation_id}")
            self._debug_print(traceback.format_exc())
            return []

    def load_latest_requirements(self, conversation_id: str) -> Dict[str, Any]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute(
                    "SELECT requirements_json FROM requirements_snapshots WHERE conversation_id = ? ORDER BY id DESC LIMIT 1",
                    (conversation_id,),
                )
                row = cur.fetchone()
                if row and row["requirements_json"]:
                    return json.loads(row["requirements_json"]) or {}
        except Exception:
            self._debug_print(f"[DB] Failed to load requirements for conversation {conversation_id}")
            self._debug_print(traceback.format_exc())
        return {}
    
    def _debug_print(self, message: str) -> None:
        if self.debug:
            print(message)

    @staticmethod
    def _truncate_for_log(text: str, max_len: int = 300) -> str:
        if text is None:
            return ""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "... [truncated]"

    def verify_ollama_connection(self):
        """Verify Ollama is running and get available models"""
        try:
            # Test if Ollama is running
            tags_url = "http://localhost:11434/api/tags"
            start_ms = time.time()
            self._debug_print(f"[LLM] Checking models at: {tags_url}")
            response = requests.get(tags_url, timeout=5)
            elapsed_ms = int((time.time() - start_ms) * 1000)
            self._debug_print(f"[LLM] GET /api/tags -> {response.status_code} in {elapsed_ms} ms")
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                print(f"✅ Ollama connected! Available models: {available_models}")
                
                # Check if our preferred model is available
                if self.model not in available_models:
                    print(f"⚠️  Model '{self.model}' not found!")
                    print(f"Available models: {available_models}")
                    
                    # Try common Qwen model names
                    qwen_models = [m for m in available_models if 'qwen' in m.lower()]
                    if qwen_models:
                        self.model = qwen_models[0]
                        print(f"🔄 Switching to: {self.model}")
                    else:
                        # Use any available model as fallback
                        if available_models:
                            self.model = available_models[0]
                            print(f"🔄 Using fallback model: {self.model}")
                        else:
                            print("❌ No models available! Please pull a model first.")
                            
            else:
                print(f"❌ Ollama connection failed: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Ollama. Is it running on localhost:11434?")
            print("   Start with: ollama serve")
        except Exception as e:
            print(f"❌ Error checking Ollama: {e}")
            self._debug_print(traceback.format_exc())
    
    def get_available_models(self):
        """Get list of available models from Ollama"""
        try:
            tags_url = "http://localhost:11434/api/tags"
            start_ms = time.time()
            self._debug_print(f"[LLM] Fetching available models: {tags_url}")
            response = requests.get(tags_url, timeout=5)
            elapsed_ms = int((time.time() - start_ms) * 1000)
            self._debug_print(f"[LLM] GET /api/tags -> {response.status_code} in {elapsed_ms} ms")
            if response.status_code == 200:
                models = response.json()
                return [model['name'] for model in models.get('models', [])]
        except:
            pass
        return []
        
    def query_ollama(self, prompt: str) -> str:
        """Query the Ollama API with the given prompt"""
        try:
            request_id = str(uuid.uuid4())[:8]
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            }
            
            self._debug_print(
                f"[LLM][{request_id}] POST {self.ollama_url} | model={self.model} | prompt_len={len(prompt)}\n"
                f"[LLM][{request_id}] Prompt preview: {self._truncate_for_log(prompt)}"
            )
            print(f"🤖 Querying model: {self.model}")

            start_ms = time.time()
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            elapsed_ms = int((time.time() - start_ms) * 1000)
            self._debug_print(
                f"[LLM][{request_id}] Response: status={response.status_code} | elapsed_ms={elapsed_ms} | bytes={len(response.text) if hasattr(response, 'text') else 'n/a'}"
            )
            
            if response.status_code == 404:
                print(f"❌ Model '{self.model}' not found. Checking available models...")
                available = self.get_available_models()
                if available:
                    print(f"Available models: {available}")
                    return f"Sorry, I'm having trouble with the AI model. Available models: {', '.join(available)}. Please check the model name in the code."
                else:
                    return f"Sorry, no AI models are available. Please install a model with: ollama pull {self.model}"
            
            response.raise_for_status()
            result = response.json()
            llm_text = (result.get('response', '') or '').strip()
            self._debug_print(
                f"[LLM][{request_id}] Parsed JSON keys: {list(result.keys())}\n"
                f"[LLM][{request_id}] Text length: {len(llm_text)} | Preview: {self._truncate_for_log(llm_text)}"
            )
            return llm_text
            
        except requests.exceptions.Timeout:
            self._debug_print(f"[LLM] Request timed out after 60s")
            return "Sorry, the AI is taking too long to respond. Please try a shorter message."
        except requests.exceptions.ConnectionError:
            self._debug_print("[LLM] Connection error to Ollama endpoint")
            return "I can't connect to the AI service. Please make sure Ollama is running (ollama serve)."
        except requests.RequestException as e:
            print(f"Error querying Ollama: {e}")
            self._debug_print(traceback.format_exc())
            return f"I'm having trouble with the AI service. Error: {str(e)}"
        except Exception as e:
            print(f"Unexpected error: {e}")
            self._debug_print(traceback.format_exc())
            return "Sorry, something went wrong. Please try again."

    def query_openrouter(self, prompt: str, model_override: str | None = None) -> str:
        """Query the OpenRouter API with the given prompt"""
        try:
            if not self.openrouter_api_key:
                return "OpenRouter API key is missing. Fill self.openrouter_api_key in code to use OpenRouter."

            request_id = str(uuid.uuid4())[:8]
            model_to_use = model_override or self.openrouter_model
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
            }
            if self.openrouter_referer:
                headers["HTTP-Referer"] = self.openrouter_referer
            if self.openrouter_title:
                headers["X-Title"] = self.openrouter_title
            payload = {
                "model": model_to_use,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
            }

            self._debug_print(
                f"[LLM][{request_id}] POST {self.openrouter_url} | model={model_to_use} | prompt_len={len(prompt)}\n"
                f"[LLM][{request_id}] Prompt preview: {self._truncate_for_log(prompt)}"
            )

            start_ms = time.time()
            response = requests.post(self.openrouter_url, headers=headers, json=payload, timeout=60)
            elapsed_ms = int((time.time() - start_ms) * 1000)
            self._debug_print(
                f"[LLM][{request_id}] Response: status={response.status_code} | elapsed_ms={elapsed_ms} | bytes={len(response.text) if hasattr(response, 'text') else 'n/a'}"
            )

            if response.status_code == 401:
                return "OpenRouter authentication failed. Check OPENROUTER_API_KEY."
            if response.status_code == 404:
                return f"OpenRouter model not found: {model_to_use}. Set OPENROUTER_MODEL or pass model in request."

            response.raise_for_status()
            result = response.json()
            # Expected: { choices: [ { message: { role, content } } ] }
            choices = result.get("choices", [])
            if not choices:
                self._debug_print(f"[LLM][{request_id}] No choices field in response: {result}")
                return "OpenRouter returned no choices."
            message = (choices[0] or {}).get("message", {})
            content = (message.get("content") or "").strip()
            self._debug_print(
                f"[LLM][{request_id}] Parsed JSON keys: {list(result.keys())}\n"
                f"[LLM][{request_id}] Text length: {len(content)} | Preview: {self._truncate_for_log(content)}"
            )
            return content

        except requests.exceptions.Timeout:
            self._debug_print("[LLM] OpenRouter request timed out after 60s")
            return "The OpenRouter request timed out. Please try again."
        except requests.RequestException as e:
            print(f"Error querying OpenRouter: {e}")
            self._debug_print(traceback.format_exc())
            try:
                err_json = response.json() if 'response' in locals() else {}
            except Exception:
                err_json = {}
            return f"OpenRouter error: {str(e)} {(' | ' + json.dumps(err_json)) if err_json else ''}"
        except Exception as e:
            print(f"Unexpected error (OpenRouter): {e}")
            self._debug_print(traceback.format_exc())
            return "Sorry, something went wrong with OpenRouter. Please try again."

    def query_llm(self, prompt: str, provider: str | None = None, model: str | None = None) -> str:
        """Unified LLM query that routes to Ollama or OpenRouter"""
        active_provider = (provider or self.provider or "ollama").lower()
        if active_provider == "openrouter":
            # If model override is from openrouter list, pass it through; otherwise fall back
            effective_model = model or getattr(self, 'openrouter_model', None)
            return self.query_openrouter(prompt, model_override=effective_model)
        # Ollama path: never silently fall back to a different local model
        if model:
            return self.query_ollama_with_model(prompt, model)
        return self.query_ollama_with_model(prompt, self.model)

    def query_ollama_with_model(self, prompt: str, model_name: str) -> str:
        previous_model = self.model
        try:
            self.model = model_name
            return self.query_ollama(prompt)
        finally:
            self.model = previous_model
    
    def extract_requirements(self, conversation_history: List[Dict], current_requirements: Dict, provider: str | None = None, model: str | None = None) -> Dict[str, Any]:
        """Extract requirements from conversation using AI"""
        
        # Build conversation context
        conversation_text = ""
        for msg in conversation_history:
            role = "Human" if msg.get('role') == 'user' else "Assistant"
            conversation_text += f"{role}: {msg.get('content', '')}\n"
        
        # Create extraction prompt
        prompt = f"""You are an expert at extracting office space requirements from conversations. 

CONVERSATION HISTORY:
{conversation_text}

CURRENT EXTRACTED REQUIREMENTS:
{json.dumps(current_requirements, indent=2)}

REQUIREMENTS TO EXTRACT (extract only if mentioned or clearly implied):
{chr(10).join([f'- {key}: {desc}' for key, desc in self.requirements_schema.items()])}

Based on the conversation, extract and return ONLY a JSON object with the requirements that can be determined from what the customer has said. Do not make assumptions beyond what is clearly stated or implied.

For numerical values, extract the actual numbers mentioned.
For descriptive values, use the customer's exact words or close paraphrases.
For yes/no questions, only answer if clearly stated.

Important: Prioritize extracting information from the most recent user message and update existing fields if the latest message clarifies them. Never remove previously captured fields; only add or refine.

Return only valid JSON with the extracted requirements:"""

        response = self.query_llm(prompt, provider=provider, model=model)
        
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
                # Merge with existing requirements, new values take precedence
                merged_requirements = {**current_requirements, **extracted}
                return merged_requirements
            else:
                return current_requirements
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error parsing requirements extraction: {e}")
            print(f"Raw response: {response}")
            return current_requirements
    
    def generate_response(self, user_message: str, conversation_history: List[Dict], current_requirements: Dict, provider: str | None = None, model: str | None = None, interview_mode: str | None = None, completion_mode: str | None = None) -> str:
        """Generate a conversational response that asks for missing information"""
        # Identify missing essentials
        missing_essentials_keys = [k for k in self.essential_fields if not current_requirements.get(k)]
        assistant_turns = sum(1 for m in conversation_history if m.get('role') == 'assistant')
        active_mode = (interview_mode or self.interview_mode).lower()
        completion_fields = self.get_completion_fields(completion_mode)

        # Auto-close when essentials are complete or too many turns
        # Close only when completion fields are filled, or safety cap reached
        all_done = all(bool(current_requirements.get(k)) for k in completion_fields)
        if self.enable_auto_close and (all_done or assistant_turns >= self.max_assistant_turns):
            return self.closing_message
            
        # Build conversation context
        conversation_text = ""
        for msg in conversation_history[-6:]:  # Last 3 exchanges
            role = "Human" if msg.get('role') == 'user' else "Assistant"
            conversation_text += f"{role}: {msg.get('content', '')}\n"
        
        # Determine prioritized fields based on mode
        prioritized_fields: List[str] = self.get_prioritized_fields(current_requirements, interview_mode)

        # Create response generation prompt
        prompt = f"""
Persona: {self.persona}

Your objectives:
- Acknowledge briefly, then ask at most {self.max_followup_questions_per_turn} focused questions that progress the search.
- Do not repeat questions already answered in CURRENT REQUIREMENTS.
- Do not claim a certain number of questions remain unless it is exactly that number based on the remaining essentials list provided.
- Keep responses concise and professional. Avoid fluff and avoid long lists.

CONVERSATION SO FAR:
{conversation_text}
Human: {user_message}

CURRENT REQUIREMENTS GATHERED:
{json.dumps(current_requirements, indent=2)}

INTERVIEW MODE: {active_mode}
FIELDS TO PRIORITIZE NOW (in order): {', '.join(prioritized_fields[:8]) if prioritized_fields else 'None'}

Your task:
1) Acknowledge briefly.
2) Ask up to {self.max_followup_questions_per_turn} questions targeting the next most important items from FIELDS TO PRIORITIZE NOW.
3) If no fields remain, ask at most one brief preference-polish question and then end.

Guidelines:
- Never repeat previously asked questions.
- Never restate the same recommendation multiple times.
- Do not mention counts unless precise and relevant; otherwise, avoid mentioning counts.

Generate a helpful, conversational response:"""

        response = self.query_llm(prompt, provider=provider, model=model)
        return response.strip()
    
    def calculate_space_recommendations(self, requirements: Dict) -> str:
        """Calculate space recommendations based on requirements"""
        if not requirements.get('employees'):
            return ""
        
        employees = int(requirements.get('employees', 0))
        business_type = requirements.get('business_type', '').lower()
        
        # Space per desk calculations
        if 'call' in business_type or 'contact' in business_type:
            sqm_per_desk = 6
            space_type = "call centre"
        elif 'professional' in business_type or 'legal' in business_type or 'accounting' in business_type:
            sqm_per_desk = 12
            space_type = "professional services"
        else:
            sqm_per_desk = 10
            space_type = "general office"
        
        # Adjust for hybrid working
        utilization = 0.6 if 'hybrid' in requirements.get('working_style', '').lower() else 1.0
        
        recommended_space = int(employees * sqm_per_desk * utilization)
        
        return f"Based on {employees} employees in {space_type} setup, I'd recommend approximately {recommended_space} sqm" + (
            " (adjusted for hybrid working)" if utilization < 1.0 else ""
        )

# Initialize the agent
agent = OfficeRequirementsAgent()

@app.route('/')
def index():
    """Serve the main HTML page"""
    # Check if index.html exists in the same directory
    if os.path.exists('index.html'):
        return send_from_directory('.', 'index.html')
    else:
        # Return a simple message with instructions
        return """
        <html>
        <head><title>Office Requirements AI</title></head>
        <body style="font-family: Arial, sans-serif; padding: 40px; text-align: center;">
            <h1>🏢 Office Requirements AI Assistant</h1>
            <div style="background: #f0f0f0; padding: 20px; border-radius: 8px; margin: 20px auto; max-width: 600px;">
                <h3>⚠️ Frontend files not found</h3>
                <p>Please make sure you have <strong>index.html</strong> and <strong>chat.js</strong> in the same directory as this Python file.</p>
                <br>
                <p><strong>Current directory:</strong> {}</p>
                <p><strong>Files found:</strong> {}</p>
            </div>
            <div style="background: #e8f4f8; padding: 15px; border-radius: 8px; margin: 20px auto; max-width: 600px;">
                <h4>🔗 API Endpoints Available:</h4>
                <ul style="text-align: left;">
                    <li><strong>GET /health</strong> - Health check</li>
                    <li><strong>POST /chat</strong> - Chat with AI</li>
                    <li><strong>GET /requirements-template</strong> - Get requirements schema</li>
                </ul>
            </div>
        </body>
        </html>
        """.format(os.getcwd(), ', '.join(os.listdir('.')))

@app.route('/chat.js')
def serve_js():
    """Serve the JavaScript file"""
    if os.path.exists('chat.js'):
        return send_from_directory('.', 'chat.js', mimetype='application/javascript')
    else:
        return "// chat.js not found", 404

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        conversation_id = data.get('conversation_id')
        user_message = data.get('message', '')
        conversation_history = data.get('conversation_history')
        current_requirements = data.get('current_requirements')
        provider_override = (data.get('provider') or None)  # 'ollama' or 'openrouter'
        model_override = (data.get('model') or None)
        interview_mode = data.get('interview_mode')  # 'core' or 'full'
        completion_mode = data.get('completion_mode')  # 'essentials' or 'full'
        force_complete = bool(data.get('force_complete'))
        # Create or load conversation
        if not conversation_id:
            conversation_id = agent.create_conversation(provider_override, model_override)
        else:
            # If overrides are provided and differ from stored values, update the conversation row
            try:
                if provider_override or model_override:
                    with sqlite3.connect(agent.db_path) as conn:
                        cur = conn.cursor()
                        if provider_override:
                            cur.execute("UPDATE conversations SET provider = ? WHERE id = ?", (provider_override, conversation_id))
                        if model_override:
                            cur.execute("UPDATE conversations SET model = ? WHERE id = ?", (model_override, conversation_id))
                        conn.commit()
            except Exception:
                pass
        if conversation_history is None:
            conversation_history = agent.load_conversation_history(conversation_id)
        if current_requirements is None:
            current_requirements = agent.load_latest_requirements(conversation_id)
        # If no overrides provided in this request, read provider/model from the conversation row
        try:
            if provider_override is None or model_override is None:
                with sqlite3.connect(agent.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cur = conn.cursor()
                    cur.execute("SELECT provider, model FROM conversations WHERE id = ? LIMIT 1", (conversation_id,))
                    row = cur.fetchone()
                    if row:
                        provider_override = provider_override or row["provider"]
                        model_override = model_override or row["model"]
        except Exception:
            pass
        
        is_complete = False

        if force_complete:
            # For development: skip LLM and force close
            assistant_response = agent.closing_message
            # Record a synthetic user action for auditability
            agent.save_message(conversation_id, 'user', user_message or '[DEV_FORCE_COMPLETE]')
            agent.save_message(conversation_id, 'assistant', assistant_response)
            updated_requirements = current_requirements or {}
            is_complete = True
        else:
            # Add user message to conversation history
            conversation_history.append({
                'role': 'user',
                'content': user_message
            })
            agent.save_message(conversation_id, 'user', user_message)

            # Extract requirements from the conversation
            updated_requirements = agent.extract_requirements(conversation_history, current_requirements, provider=provider_override, model=model_override)
            agent.save_requirements_snapshot(conversation_id, updated_requirements)

            # Generate response
            assistant_response = agent.generate_response(
                user_message,
                conversation_history,
                updated_requirements,
                provider=provider_override,
                model=model_override,
                interview_mode=interview_mode,
                completion_mode=completion_mode,
            )
            agent.save_message(conversation_id, 'assistant', assistant_response)
        
        # Optionally add a single recommendation once (less chatty)
        if '💡' not in ''.join([m.get('content','') for m in conversation_history]):
            space_rec = agent.calculate_space_recommendations(updated_requirements)
            if space_rec and space_rec not in assistant_response:
                assistant_response += f"\n\n💡 {space_rec}"
        
        # Add assistant response to conversation history
        conversation_history.append({
            'role': 'assistant',
            'content': assistant_response
        })

        # Completion detection flag for frontend
        try:
            completion_fields = agent.get_completion_fields(completion_mode)
            have_all = all(bool(updated_requirements.get(k)) for k in completion_fields)
            is_complete = is_complete or have_all or (assistant_response.strip() == agent.closing_message.strip())
        except Exception:
            pass
        
        # Reflect effective provider/model back to client
        effective_provider = (provider_override or agent.provider)
        if effective_provider == 'openrouter':
            effective_model = (model_override or agent.openrouter_model)
        else:
            effective_model = (model_override or agent.model)

        return jsonify({
            'success': True,
            'reply': assistant_response,
            'requirements': updated_requirements,
            'conversation_history': conversation_history,
            'conversation_id': conversation_id,
            'provider': effective_provider,
            'model': effective_model,
            'interview_mode': interview_mode or agent.interview_mode,
            'completion_mode': completion_mode or agent.completion_mode,
            'is_complete': bool(is_complete)
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        available_models = []
        ollama_status = "disabled"
        if agent.provider == 'ollama':
            # Test connection to Ollama
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                ollama_status = "connected"
            else:
                available_models = []
                ollama_status = "disconnected"
    except Exception as e:
        available_models = []
        ollama_status = f"error: {str(e)}"
    
    return jsonify({
        'status': 'running',
        'provider': agent.provider,
        'ollama': ollama_status,
        'current_model': agent.model if agent.provider == 'ollama' else agent.openrouter_model,
        'available_models': available_models,
        'openrouter_models': getattr(agent, 'openrouter_models', [agent.openrouter_model])
    })

@app.route('/requirements-template', methods=['GET'])
def get_requirements_template():
    """Return the full requirements schema for reference"""
    return jsonify({
        'schema': agent.requirements_schema,
        'total_fields': len(agent.requirements_schema)
    })

# --------- Models listing (dev) ---------
@app.route('/models', methods=['GET'])
def list_models():
    """Return available models for both providers regardless of current provider setting."""
    # OpenRouter list from agent config
    openrouter_models = getattr(agent, 'openrouter_models', [getattr(agent, 'openrouter_model', '')])
    if not isinstance(openrouter_models, list):
        openrouter_models = [str(openrouter_models)]

    # Ollama models by querying local API; safe fail to []
    ollama_models: List[str] = []
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json() or {}
            ollama_models = [m.get('name') for m in models.get('models', []) if m.get('name')]
    except Exception:
        ollama_models = []

    return jsonify({
        'openrouter_models': openrouter_models,
        'ollama_models': ollama_models,
    })

# -------------------- DEV EXPLORER ENDPOINTS --------------------
@app.route('/dev/conversations', methods=['GET'])
def dev_list_conversations():
    try:
        limit = int(request.args.get('limit', '100'))
        offset = int(request.args.get('offset', '0'))

        items: List[Dict[str, Any]] = []
        total = 0
        with sqlite3.connect(agent.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # total count
            cur.execute("SELECT COUNT(*) as c FROM conversations")
            total = int((cur.fetchone() or {"c": 0})["c"])

            # base list ordered by created_at DESC
            cur.execute(
                "SELECT id, created_at, provider, model FROM conversations ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = cur.fetchall()

            for row in rows:
                conversation_id = row["id"]
                # message count and last timestamp
                cur.execute("SELECT COUNT(*) as c, MAX(created_at) as last_at FROM messages WHERE conversation_id = ?", (conversation_id,))
                m = cur.fetchone() or {"c": 0, "last_at": None}

                # first user message preview as title
                cur.execute(
                    "SELECT content FROM messages WHERE conversation_id = ? AND role = 'user' ORDER BY id ASC LIMIT 1",
                    (conversation_id,),
                )
                first_user = cur.fetchone()
                title = (first_user["content"] if first_user else "Untitled").strip()
                if len(title) > 60:
                    title = title[:57] + "…"

                # last assistant message for completion check
                cur.execute(
                    "SELECT content FROM messages WHERE conversation_id = ? AND role = 'assistant' ORDER BY id DESC LIMIT 1",
                    (conversation_id,),
                )
                last_asst_row = cur.fetchone()
                last_asst_text = (last_asst_row["content"] if last_asst_row else "")
                # fetch latest requirements snapshot
                cur.execute(
                    "SELECT requirements_json FROM requirements_snapshots WHERE conversation_id = ? ORDER BY id DESC LIMIT 1",
                    (conversation_id,),
                )
                req_row = cur.fetchone()
                latest_requirements = {}
                if req_row and req_row["requirements_json"]:
                    try:
                        latest_requirements = json.loads(req_row["requirements_json"]) or {}
                    except Exception:
                        latest_requirements = {}

                # compute completion and filled counts
                completion_fields = agent.get_completion_fields(None)
                have_all = all(bool(latest_requirements.get(k)) for k in completion_fields)
                is_complete = bool(have_all or (last_asst_text.strip() == agent.closing_message.strip()))

                def _count_non_empty(d: dict) -> int:
                    count = 0
                    for v in (d or {}).values():
                        if v is None:
                            continue
                        if isinstance(v, str):
                            if v.strip():
                                count += 1
                        else:
                            # numbers/bools/objects considered filled if truthy
                            if bool(v):
                                count += 1
                    return count

                filled_fields_count = _count_non_empty(latest_requirements)
                filled_fields_total = len(agent.get_all_section_fields())

                items.append({
                    "id": conversation_id,
                    "created_at": row["created_at"],
                    "provider": row["provider"],
                    "model": row["model"],
                    "message_count": int(m["c"] if isinstance(m, sqlite3.Row) else (m[0] if isinstance(m, (list, tuple)) else 0)),
                    "last_message_at": m["last_at"] if isinstance(m, sqlite3.Row) else None,
                    "title": title,
                    "is_complete": is_complete,
                    "filled_fields_count": filled_fields_count,
                    "filled_fields_total": filled_fields_total,
                })

        return jsonify({"items": items, "total": total})
    except Exception as e:
        agent._debug_print(f"[DEV] Error listing conversations: {e}")
        agent._debug_print(traceback.format_exc())
        return jsonify({"items": [], "total": 0, "error": str(e)}), 500


@app.route('/dev/conversations/<conversation_id>', methods=['GET'])
def dev_get_conversation(conversation_id: str):
    try:
        with sqlite3.connect(agent.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # conversation core
            cur.execute("SELECT id, created_at, provider, model FROM conversations WHERE id = ?", (conversation_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "not_found"}), 404

            # messages
            cur.execute("SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC", (conversation_id,))
            messages = [
                {"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
                for r in cur.fetchall()
            ]

            # latest requirements
            cur.execute(
                "SELECT requirements_json FROM requirements_snapshots WHERE conversation_id = ? ORDER BY id DESC LIMIT 1",
                (conversation_id,),
            )
            req_row = cur.fetchone()
            requirements_latest = {}
            if req_row and req_row["requirements_json"]:
                try:
                    requirements_latest = json.loads(req_row["requirements_json"]) or {}
                except Exception:
                    requirements_latest = {}

            completion_fields = agent.get_completion_fields(None)
            have_all = all(bool(requirements_latest.get(k)) for k in completion_fields)
            # last assistant message
            cur.execute(
                "SELECT content FROM messages WHERE conversation_id = ? AND role = 'assistant' ORDER BY id DESC LIMIT 1",
                (conversation_id,),
            )
            last_asst_row = cur.fetchone()
            last_asst_text = (last_asst_row["content"] if last_asst_row else "")
            is_complete = bool(have_all or (last_asst_text.strip() == agent.closing_message.strip()))

            def _count_non_empty(d: dict) -> int:
                count = 0
                for v in (d or {}).values():
                    if v is None:
                        continue
                    if isinstance(v, str):
                        if v.strip():
                            count += 1
                    else:
                        if bool(v):
                            count += 1
                return count

            filled_fields_count = _count_non_empty(requirements_latest)
            filled_fields_total = len(agent.get_all_section_fields())

            return jsonify({
                "id": row["id"],
                "created_at": row["created_at"],
                "provider": row["provider"],
                "model": row["model"],
                "is_complete": is_complete,
                "filled_fields_count": filled_fields_count,
                "filled_fields_total": filled_fields_total,
                "requirements_latest": requirements_latest,
                "messages": messages,
            })
    except Exception as e:
        agent._debug_print(f"[DEV] Error getting conversation {conversation_id}: {e}")
        agent._debug_print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/dev/conversations/<conversation_id>', methods=['DELETE'])
def dev_delete_conversation(conversation_id: str):
    """Delete a conversation and all related rows (messages, requirements_snapshots)."""
    try:
        with sqlite3.connect(agent.db_path) as conn:
            cur = conn.cursor()
            # Check existence first
            cur.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,))
            if not cur.fetchone():
                return jsonify({"success": False, "error": "not_found"}), 404

            # Delete related rows then the conversation
            cur.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            cur.execute("DELETE FROM requirements_snapshots WHERE conversation_id = ?", (conversation_id,))
            cur.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
        return jsonify({"success": True})
    except Exception as e:
        agent._debug_print(f"[DEV] Error deleting conversation {conversation_id}: {e}")
        agent._debug_print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("🏢 Office Requirements AI Assistant")
    print("=" * 50)
    if agent.provider == 'openrouter':
        print(f"🤖 Provider: OpenRouter | Model: {agent.openrouter_model}")
        print("🔑 OPENROUTER_API_KEY is " + ("set" if bool(agent.openrouter_api_key) else "NOT set"))
    else:
        print(f"🤖 Provider: Ollama | Model: {agent.model}")
    print("🌐 Starting server on http://localhost:5000")
    print("📋 Health check: http://localhost:5000/health")
    if agent.provider == 'ollama':
        print(f"\nMake sure Ollama is running and the model is available: {agent.model}")
        print(f"Suggested pull: ollama pull {agent.model}")
        print(f"Run: ollama run {agent.model}")
    else:
        print("\nUsing OpenRouter. Ensure OPENROUTER_API_KEY is configured.")
        print(f"Endpoint: {agent.openrouter_url}")
        print(f"Models: {getattr(agent, 'openrouter_models', [agent.openrouter_model])}")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
