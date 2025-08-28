import os
import base64
import re
import json
import time

import streamlit as st
import openai
from openai import AssistantEventHandler
from tools import TOOL_MAP
from typing_extensions import override
from dotenv import load_dotenv
import streamlit_authenticator as stauth

# Load .env only if running locally
if os.path.exists(".env"):
    load_dotenv()

def str_to_bool(str_input):
    if not isinstance(str_input, str):
        return False
    return str_input.lower() == "true"


# ✅ Load environment variables with fallback to st.secrets
def get_secret(key: str, default=None):
    if key in st.secrets:   # Streamlit Cloud secrets
        return st.secrets[key]
    return os.environ.get(key, default)  # Local .env


# ---- Configuration ----
openai_api_key = get_secret("OPENAI_API_KEY")
instructions = get_secret("RUN_INSTRUCTIONS", "")
enabled_file_upload_message = get_secret("ENABLED_FILE_UPLOAD_MESSAGE", "Upload a file")
azure_openai_endpoint = get_secret("AZURE_OPENAI_ENDPOINT")
azure_openai_key = get_secret("AZURE_OPENAI_KEY")
authentication_required = str_to_bool(get_secret("AUTHENTICATION_REQUIRED", "False"))

USER_INSTRUCTIONS = """
You are my Italian law assistant. Provide:
- Complete deeds in .docx when I request “dammi un atto [type]”.
- Explanations of any clause when I ask “spiegami la clausola [X] dell’atto di [type]”.
- If something changes in external data, update the answer accordingly.
USER_INSTRUCTIONS
Example Queries:
- “Dammi un atto di costituzione società tipo SPA”
- “Spiegami la clausola di responsabilità nell’atto di SPA”
- “Quali documenti servono per registrare una SRL in Italia?”

"""

# Load authentication configuration
if authentication_required:
    if "credentials" in st.secrets:
        authenticator = stauth.Authenticate(
            st.secrets["credentials"].to_dict(),
            st.secrets["cookie"]["name"],
            st.secrets["cookie"]["key"],
            st.secrets["cookie"]["expiry_days"],
        )
    else:
        authenticator = None

client = None
if azure_openai_endpoint and azure_openai_key:
    client = openai.AzureOpenAI(
        api_key=azure_openai_key,
        api_version="2024-05-01-preview",
        azure_endpoint=azure_openai_endpoint,
    )
else:
    client = openai.OpenAI(api_key=openai_api_key)


class EventHandler(AssistantEventHandler):
    def __init__(self):
        super().__init__()

    @override
    def on_event(self, event):
        # Add debugging
        print(f"Event received: {event.event}")
        if hasattr(event, 'data') and hasattr(event.data, 'status'):
            print(f"Run status: {event.data.status}")

    @override
    def on_run_step_created(self, run_step):
        print(f"Run step created: {run_step.type}")
        
    @override
    def on_run_step_done(self, run_step):
        print(f"Run step completed: {run_step.type}")

    @override
    def on_run_step_failed(self, run_step):
        print(f"Run step failed: {run_step.type}")
        if run_step.last_error:
            print(f"Error: {run_step.last_error}")
            st.error(f"Run step failed: {run_step.last_error}")

    @override 
    def on_run_completed(self, run):
        print(f"Run completed: {run.id}")
        
    @override
    def on_run_failed(self, run):
        print(f"Run failed: {run.id}")
        if run.last_error:
            print(f"Run error: {run.last_error}")
            st.error(f"Run failed: {run.last_error}")
            
    @override
    def on_run_cancelled(self, run):
        print(f"Run cancelled: {run.id}")
        
    @override
    def on_run_expired(self, run):
        print(f"Run expired: {run.id}")
        st.error("The assistant run has expired. Please try again.")

    @override
    def on_text_created(self, text):
        print("Text created")
        st.session_state.current_message = ""
        with st.chat_message("Assistant"):
            st.session_state.current_markdown = st.empty()

    @override
    def on_text_delta(self, delta, snapshot):
        print("Text delta: ", delta.value if delta.value else "None")
        if snapshot.value:
            text_value = re.sub(
                r"\[(.*?)\]\s*\(\s*(.*?)\s*\)", "Download Link", snapshot.value
            )
            st.session_state.current_message = text_value
            st.session_state.current_markdown.markdown(
                st.session_state.current_message, True
            )

    @override
    def on_text_done(self, text):
        print("Text done: ", text.value if text else "None")
        format_text = format_annotation(text)
        st.session_state.current_markdown.markdown(format_text, True)
        st.session_state.chat_log.append({"name": "assistant", "msg": format_text})

    @override
    def on_tool_call_created(self, tool_call):
        print("Tool call created: ", tool_call.type)
        if tool_call.type == "code_interpreter":
            st.session_state.current_tool_input = ""
            with st.chat_message("Assistant"):
                st.session_state.current_tool_input_markdown = st.empty()

    @override
    def on_tool_call_delta(self, delta, snapshot):
        if 'current_tool_input_markdown' not in st.session_state:
            with st.chat_message("Assistant"):
                st.session_state.current_tool_input_markdown = st.empty()

        if delta.type == "code_interpreter":
            if delta.code_interpreter.input:
                st.session_state.current_tool_input += delta.code_interpreter.input
                input_code = f"### code interpreter\ninput:\n```python\n{st.session_state.current_tool_input}\n```"
                st.session_state.current_tool_input_markdown.markdown(input_code, True)

    @override
    def on_tool_call_done(self, tool_call):
        print(f"Tool call done: {tool_call.type}")
        
        # Prevent duplicate processing
        if tool_call.id in [x.id for x in st.session_state.tool_calls]:
            return
            
        st.session_state.tool_calls.append(tool_call)
        
        if tool_call.type == "code_interpreter":
            input_code = f"### code interpreter\ninput:\n```python\n{tool_call.code_interpreter.input}\n```"
            if 'current_tool_input_markdown' in st.session_state and st.session_state.current_tool_input_markdown:
                st.session_state.current_tool_input_markdown.markdown(input_code, True)
            st.session_state.chat_log.append({"name": "assistant", "msg": input_code})
            
            for output in tool_call.code_interpreter.outputs:
                if output.type == "logs":
                    output_text = f"### code interpreter\noutput:\n```\n{output.logs}\n```"
                    with st.chat_message("Assistant"):
                        st.markdown(output_text, True)
                        st.session_state.chat_log.append(
                            {"name": "assistant", "msg": output_text}
                        )
                elif output.type == "image":
                    # Handle image outputs
                    with st.chat_message("Assistant"):
                        st.image(f"data:image/png;base64,{output.image.file_id}")
                        
        elif tool_call.type == "function":
            # Function tool calls are handled via on_run_requires_action.
            # Do not execute tools here to avoid duplicate runs.
            print(
                f"Function tool call observed (id={tool_call.id}, name={tool_call.function.name}) — awaiting requires_action to submit outputs."
            )

    @override
    def on_run_requires_action(self, run, run_step):
        """Handle when the run requires action (function calls)"""
        print(f"Run requires action: {run.id}")
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        tool_outputs = []
        
        for tool_call in tool_calls:
            try:
                tool_function_name = tool_call.function.name
                tool_function_arguments = json.loads(tool_call.function.arguments)
                
                with st.chat_message("Assistant"):
                    msg = f"### Function Calling: {tool_function_name}"
                    st.markdown(msg, True)
                    st.session_state.chat_log.append({"name": "assistant", "msg": msg})
                
                if tool_function_name in TOOL_MAP:
                    tool_function_output = TOOL_MAP[tool_function_name](**tool_function_arguments)
                    out = str(tool_function_output) if tool_function_output is not None else ""
                    if len(out) > 100000:
                        out = out[:100000] + "\n... [truncated]"
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": out,
                    })
                else:
                    print(f"Unknown function: {tool_function_name}")
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": f"Error: Unknown function {tool_function_name}",
                    })
                    
            except Exception as e:
                print(f"Error executing function {tool_call.function.name}: {e}")
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": f"Error: {str(e)}",
                })
        
        # Submit tool outputs (non-stream) to avoid nested streaming in event handler
        if tool_outputs:
            try:
                # Show a short preview in the UI for transparency/debugging
                with st.chat_message("Assistant"):
                    preview = "\n\n".join(
                        [
                            f"Submitted tool output for {tc.function.name}:\n```${out['output'][:500]}...```"
                            for tc, out in zip(tool_calls, tool_outputs)
                        ]
                    )
                    st.info(preview)

                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=st.session_state.thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs,
                )
            except Exception as e:
                print(f"Error submitting tool outputs: {e}")

    def handle_function_call(self, tool_call):
        """Handle function tool calls - deprecated in favor of on_run_requires_action"""
        print(
            f"Function call will be handled in on_run_requires_action: {tool_call.function.name}"
        )

    @override
    def on_exception(self, exception):
        print(f"Exception occurred: {exception}")
        st.error(f"An error occurred: {str(exception)}")


def create_thread(content, file):
    """Create a new thread"""
    try:
        return client.beta.threads.create()
    except Exception as e:
        print(f"Error creating thread: {e}")
        st.error(f"Error creating thread: {str(e)}")
        return None


def create_message(thread, content, file):
    """Create a message in the thread"""
    try:
        attachments = []
        if file is not None:
            attachments.append({
                "file_id": file.id, 
                "tools": [{"type": "code_interpreter"}, {"type": "file_search"}]
            })
            
        message = client.beta.threads.messages.create(
            thread_id=thread.id, 
            role="user", 
            content=content, 
            attachments=attachments
        )
        print(f"Message created: {message.id}")
        return message
    except Exception as e:
        print(f"Error creating message: {e}")
        st.error(f"Error creating message: {str(e)}")
        return None


def create_file_link(file_name, file_id):
    """Create downloadable file link"""
    try:
        content = client.files.content(file_id)
        content_type = content.response.headers["content-type"]
        b64 = base64.b64encode(content.text.encode(content.encoding)).decode()
        link_tag = f'<a href="data:{content_type};base64,{b64}" download="{file_name}">Download Link</a>'
        return link_tag
    except Exception as e:
        print(f"Error creating file link: {e}")
        return "Download Link"


def format_annotation(text):
    """Format text with annotations and citations"""
    try:
        citations = []
        text_value = text.value
        
        for index, annotation in enumerate(text.annotations):
            text_value = text_value.replace(annotation.text, f" [{index}]")

            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = client.files.retrieve(file_citation.file_id)
                citations.append(
                    f"[{index}] {file_citation.quote} from {cited_file.filename}"
                )
            elif file_path := getattr(annotation, "file_path", None):
                link_tag = create_file_link(
                    annotation.text.split("/")[-1],
                    file_path.file_id,
                )
                text_value = re.sub(r"\[(.*?)\]\s*\(\s*(.*?)\s*\)", link_tag, text_value)
                
        if citations:
            text_value += "\n\n" + "\n".join(citations)
        return text_value
    except Exception as e:
        print(f"Error formatting annotation: {e}")
        return text.value if text else ""


def run_stream(user_input, file, selected_assistant_id):
    """Run the assistant with streaming"""
    try:
        # Create thread if it doesn't exist
        if "thread" not in st.session_state or st.session_state.thread is None:
            st.session_state.thread = create_thread(user_input, file)
            if st.session_state.thread is None:
                return
        
        # Create message
        message = create_message(st.session_state.thread, user_input, file)
        if message is None:
            return
            
        # Verify assistant exists
        try:
            assistant = client.beta.assistants.retrieve(selected_assistant_id)
            print(f"Using assistant: {assistant.name}")
        except Exception as e:
            st.error(f"Assistant not found: {selected_assistant_id}")
            print(f"Assistant error: {e}")
            return
        
        # Create event handler
        event_handler = EventHandler()
        
        # Start streaming run
        print(f"Starting run with assistant: {selected_assistant_id}")
        with client.beta.threads.runs.stream(
            thread_id=st.session_state.thread.id,
            assistant_id=selected_assistant_id,
            event_handler=event_handler,
            instructions=USER_INSTRUCTIONS if USER_INSTRUCTIONS else None
        ) as stream:
            stream.until_done()
            
    except Exception as e:
        print(f"Error in run_stream: {e}")
        st.error(f"Error running assistant: {str(e)}")


def handle_uploaded_file(uploaded_file):
    """Handle file upload to OpenAI"""
    try:
        file = client.files.create(file=uploaded_file, purpose="assistants")
        print(f"Upload file: {file.id}")
        return file
    except Exception as e:
        print(f"Error uploading file: {e}")
        st.error(f"Error uploading file: {str(e)}")
        return None


def render_chat():
    """Render chat messages"""
    for chat in st.session_state.chat_log:
        with st.chat_message(chat["name"]):
            st.markdown(chat["msg"], True)


# Initialize session state
if "tool_calls" not in st.session_state:
    st.session_state.tool_calls = []

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

if "in_progress" not in st.session_state:
    st.session_state.in_progress = False

if "thread" not in st.session_state:
    st.session_state.thread = None


def disable_form():
    st.session_state.in_progress = True


def login():
    if st.session_state["authentication_status"] is False:
        st.error("Username/password is incorrect")
    elif st.session_state["authentication_status"] is None:
        st.warning("Please enter your username and password")


def reset_chat():
    """Reset chat and create new thread"""
    st.session_state.chat_log = []
    st.session_state.in_progress = False
    st.session_state.thread = None
    st.session_state.tool_calls = []


def load_chat_screen(assistant_id, assistant_title):
    """Load the main chat interface"""
    # File uploader
    uploaded_file = None
    if enabled_file_upload_message:
        uploaded_file = st.sidebar.file_uploader(
            enabled_file_upload_message,
            type=["txt", "pdf", "csv", "json", "geojson", "xlsx", "xls"],
            disabled=st.session_state.in_progress,
        )

    # Reset chat button
    if st.sidebar.button("Reset Chat"):
        reset_chat()
        st.rerun()

    st.title(assistant_title if assistant_title else "Assistant")
    
    # Chat input
    user_msg = st.chat_input(
        "Message", on_submit=disable_form, disabled=st.session_state.in_progress
    )
    
    if user_msg:
        render_chat()
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_msg, True)
        st.session_state.chat_log.append({"name": "user", "msg": user_msg})

        # Handle file upload
        file = None
        if uploaded_file is not None:
            file = handle_uploaded_file(uploaded_file)
            if file:
                with st.chat_message("user"):
                    st.info(f"File uploaded: {uploaded_file.name}")
        
        # Run assistant
        run_stream(user_msg, file, assistant_id)
        
        # Reset progress state
        st.session_state.in_progress = False
        st.rerun()

    render_chat()


def main():
    """Main application function"""
    # Check configuration
    multi_agents = os.environ.get("OPENAI_ASSISTANTS", None)
    single_agent_id = os.environ.get("ASSISTANT_ID", None)
    single_agent_title = os.environ.get("ASSISTANT_TITLE", "Assistants API UI")

    # Handle authentication
    if (
        authentication_required
        and "credentials" in st.secrets
        and authenticator is not None
    ):
        authenticator.login()
        if not st.session_state["authentication_status"]:
            login()
            return
        else:
            authenticator.logout(location="sidebar")

    # Load assistant configuration
    if multi_agents:
        try:
            assistants_json = json.loads(multi_agents)
            assistants_object = {f'{obj["title"]}': obj for obj in assistants_json}
            selected_assistant = st.sidebar.selectbox(
                "Select an assistant profile?",
                list(assistants_object.keys()),
                index=None,
                placeholder="Select an assistant profile...",
                on_change=reset_chat,
            )
            if selected_assistant:
                load_chat_screen(
                    assistants_object[selected_assistant]["id"],
                    assistants_object[selected_assistant]["title"],
                )
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON in OPENAI_ASSISTANTS: {e}")
    elif single_agent_id:
        load_chat_screen(single_agent_id, single_agent_title)
    else:
        st.error("No assistant configurations defined in environment variables.")


if __name__ == "__main__":
    main()