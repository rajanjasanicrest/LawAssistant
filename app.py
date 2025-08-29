import os
import base64
import re
import json
import time
import urllib.parse

import streamlit as st
import openai
from openai import AssistantEventHandler
from tools import TOOL_MAP
from typing_extensions import override
from dotenv import load_dotenv
import streamlit_authenticator as stauth

load_dotenv()


def str_to_bool(str_input):
    if not isinstance(str_input, str):
        return False
    return str_input.lower() == "true"


# Load environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")
instructions = os.environ.get("RUN_INSTRUCTIONS", "")
enabled_file_upload_message = os.environ.get(
    "ENABLED_FILE_UPLOAD_MESSAGE", "Upload a file"
)
azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.environ.get("AZURE_OPENAI_KEY")
authentication_required = str_to_bool(os.environ.get("AUTHENTICATION_REQUIRED", False))

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


def create_thread(content):
    """Create a new thread"""
    try:
        return client.beta.threads.create()
    except Exception as e:
        print(f"Error creating thread: {e}")
        st.error(f"Error creating thread: {str(e)}")
        return None


def create_message(thread, content):
    """Create a message in the thread"""
    try:
        message = client.beta.threads.messages.create(
            thread_id=thread.id, 
            role="user", 
            content=content,
        )
        print(f"Message created: {message.id}")
        return message
    except Exception as e:
        print(f"Error creating message: {e}")
        st.error(f"Error creating message: {str(e)}")
        return None


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
                
        if citations:
            text_value += "\n\n" + "\n".join(citations)
        return text_value
    except Exception as e:
        print(f"Error formatting annotation: {e}")
        return text.value if text else ""


def extract_download_info(text):
    """Extract download information from text"""
    # More comprehensive regex patterns including markdown links
    # patterns = [
    #     # Markdown link format: [text](url)
    #     r"\[.*?\]\((https?://[^\s\)]+?\.docx(?:\?[^\s\)]*)?)\)",
    #     # Direct URLs
    #     r"https?://[^\s]+?\.docx(?:\?[^\s]*)?",
    #     r"http://[^\s]+?/templates/[^\s]+?\.docx",
    #     r"http://localhost:\d+/templates/[^\s]+?\.docx",
    #     # Streamlit app URLs
    #     r"https?://[^\s]*?streamlit\.app[^\s]*?/templates/[^\s]+?\.docx",
    # ]
    patterns = [
        # Markdown link format: [text](url) - captures everything inside parentheses
        r"\[.*?\]\((https?://.*?\.docx(?:\?[^)]*)?)\)",
        # Direct URLs ending with .docx - captures until whitespace that's not part of filename
        r"https?://\S*?[^\s]*\.docx(?:\?[^\s]*)?(?=\s|$|[.,;!?])",
        # URLs with spaces - captures everything until common delimiters
        r"https?://[^\n\r\t]*?\.docx(?:\?[^\n\r\t]*)?(?=\s*$|\s*[.,;!?\n\r\t]|\s+[A-Za-z])",
        # Template URLs with potential spaces
        r"https?://[^\n\r\t]*?/templates/[^\n\r\t]*?\.docx(?:\?[^\n\r\t]*)?(?=\s*$|\s*[.,;!?\n\r\t]|\s+[A-Za-z])",
        # Localhost URLs with potential spaces
        r"http://localhost:\d+/templates/[^\n\r\t]*?\.docx(?:\?[^\n\r\t]*)?(?=\s*$|\s*[.,;!?\n\r\t]|\s+[A-Za-z])",
        # Streamlit app URLs with potential spaces
        r"https?://[^\n\r\t]*?streamlit\.app[^\n\r\t]*?/templates/[^\n\r\t]*?\.docx(?:\?[^\n\r\t]*)?(?=\s*$|\s*[.,;!?\n\r\t]|\s+[A-Za-z])",
    ]
    
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Handle both direct matches and group matches from markdown
            url = matches[0] if isinstance(matches[0], str) else matches[0]
            print(f"Found URL: {url}")
            
            # Extract filename from URL
            filename = url.split("/")[-1].split("?")[0]  # Remove query params
            filename = urllib.parse.unquote(filename)
            print(f"Extracted filename: {filename}")
            
            # Check if file exists locally
            file_path = f'templates/{filename}'
            print(f"Checking file at: {file_path}")
            
            if os.path.exists(file_path):
                print(f"File found at: {file_path}")
                return {
                    'file_path': file_path,
                    'filename': filename,
                    'url': url
                }
            else:
                print(f"File not found at: {file_path}")
                # Still return the info even if file doesn't exist locally
                # The assistant might have generated it
                return {
                    'file_path': file_path,
                    'filename': filename,
                    'url': url
                }
                
    print("No download URLs found in text")
    return None


def run_assistant_stream(user_input, selected_assistant_id):
    """Run the assistant with proper state management"""
    try:
        # Set processing state
        st.session_state.in_progress = True
        
        # Create thread if it doesn't exist
        if "thread" not in st.session_state or st.session_state.thread is None:
            st.session_state.thread = create_thread(user_input)
            if st.session_state.thread is None:
                st.session_state.in_progress = False
                return
        
        # Create message
        message = create_message(st.session_state.thread, user_input)
        if message is None:
            st.session_state.in_progress = False
            return
            
        # Verify assistant exists
        try:
            assistant = client.beta.assistants.retrieve(selected_assistant_id)
            print(f"Using assistant: {assistant.name}")
        except Exception as e:
            st.error(f"Assistant not found: {selected_assistant_id}")
            print(f"Assistant error: {e}")
            st.session_state.in_progress = False
            return
        
        # Start run
        print(f"Starting run with assistant: {selected_assistant_id}")
        run = client.beta.threads.runs.create(
            thread_id=st.session_state.thread.id,
            assistant_id=selected_assistant_id,
        )
        
        max_iterations = 60
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get current run status
            run = client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread.id,
                run_id=run.id
            )
            print(f"Run status: {run.status} (iteration {iteration})")
            
            if run.status == "completed":
                print("✅ Run completed successfully!")
                break
                
            elif run.status == "requires_action":
                print("🔧 Function calling required!")
                
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    
                    print(f"  Calling: {function_name}({function_args})")
                    
                    if function_name in TOOL_MAP:
                        try:
                            output = TOOL_MAP[function_name](**function_args)
                            print(f"  Result: {output}")
                            
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": output
                            })
                        except Exception as e:
                            print(f"  Error: {e}")
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": f"Error: {e}"
                            })
                    else:
                        print(f"  Unknown function: {function_name}")
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": f"Error: Unknown function {function_name}"
                        })
                
                # Submit the outputs
                print(f"📤 Submitting {len(tool_outputs)} tool outputs...")
                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=st.session_state.thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
                print("✅ Tool outputs submitted")
                
            elif run.status in ["failed", "cancelled", "expired"]:
                print(f"❌ Run {run.status}")
                if run.last_error:
                    print(f"Error: {run.last_error}")
                    st.error(f"Run {run.status}: {run.last_error}")
                break
                
            elif run.status in ["queued", "in_progress"]:
                time.sleep(1)
            else:
                print(f"Unknown status: {run.status}")
                break
        
        if iteration >= max_iterations:
            print("❌ Run timed out")
            st.error("The request timed out. Please try again.")
            st.session_state.in_progress = False
            return
            
        # Get the final messages and display them
        time.sleep(0.5)
        
        messages = client.beta.threads.messages.list(thread_id=st.session_state.thread.id)
        
        # Process the latest assistant message
        for message in messages.data:
            if message.role == "assistant" and message.created_at > (time.time() - 300):
                for content in message.content:
                    if content.type == "text":
                        formatted_text = format_annotation(content.text)
                        
                        # Replace URLs in the text for display
                        display_text = formatted_text.replace(
                            'https://example.com/', 
                            os.environ.get("FILE_SERVER_BASE_URL", "http://localhost:8501/templates/")
                        )
                        
                        # Check for download info
                        download_info = extract_download_info(display_text)
                        if download_info:
                            st.session_state.download_info = download_info
                            print(f"Download info stored: {download_info['filename']}")
                        
                        # Add to chat log (this will be rendered by render_chat)
                        st.session_state.chat_log.append({"name": "assistant", "msg": display_text})
                        
                break  # Only process the latest message
                
    except Exception as e:
        print(f"Error in run_assistant_stream: {e}")
        st.error(f"Error running assistant: {str(e)}")
    finally:
        # Always reset processing state
        st.session_state.in_progress = False


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
            st.markdown(chat["msg"], unsafe_allow_html=True)


def render_download_button():
    """Render download button if available"""
    if "download_info" in st.session_state and st.session_state.download_info:
        info = st.session_state.download_info
        try:
            with open(info['file_path'], "rb") as f:
                file_bytes = f.read()

            st.markdown("---")
            st.markdown("## 📥 Your document is ready:")
            st.download_button(
                label="📄 Download Template",
                data=file_bytes,
                file_name=info['filename'],
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"download_{info['filename']}_{len(st.session_state.chat_log)}"
            )
        except Exception as e:
            st.error(f"Error creating download button: {e}")


def reset_chat():
    """Reset chat and create new thread"""
    st.session_state.chat_log = []
    st.session_state.in_progress = False
    st.session_state.thread = None
    if "download_info" in st.session_state:
        del st.session_state.download_info


def load_chat_screen(assistant_id, assistant_title):
    """Load the main chat interface"""
    
    # Initialize session state
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []
    if "in_progress" not in st.session_state:
        st.session_state.in_progress = False
    if "thread" not in st.session_state:
        st.session_state.thread = None

    # Reset chat button in sidebar
    if st.sidebar.button("Reset Chat"):
        reset_chat()
        st.rerun()

    st.title(assistant_title if assistant_title else "Italian Law Assistant")

    # Render existing chat messages
    render_chat()
    
    # Render download button if available
    render_download_button()

    # Chat input - process immediately when submitted
    user_msg = st.chat_input(
        "Ask me about legal documents...", 
        disabled=st.session_state.in_progress
    )
    
    # Handle new user message
    if user_msg and not st.session_state.in_progress:
        # Add user message to chat log and show immediately
        st.session_state.chat_log.append({"name": "user", "msg": user_msg})
        
        # Show user message immediately
        with st.chat_message("user"):
            st.session_state.download_info = None  # Clear previous download info
            st.markdown(user_msg, True)
        
        # Process with assistant
        with st.spinner("Processing your request..."):
            run_assistant_stream(user_msg, assistant_id)
        
        # Rerun to update UI with assistant response
        st.rerun()


def login():
    if st.session_state["authentication_status"] is False:
        st.error("Username/password is incorrect")
    elif st.session_state["authentication_status"] is None:
        st.warning("Please enter your username and password")


def main():
    """Main application function"""
    # Check configuration
    multi_agents = os.environ.get("OPENAI_ASSISTANTS", None)
    single_agent_id = os.environ.get("ASSISTANT_ID", None)
    single_agent_title = os.environ.get("ASSISTANT_TITLE", "Assistants API UI")

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