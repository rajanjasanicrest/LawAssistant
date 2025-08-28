import os
import base64
import re
import json
import time
import urllib.parse  # Add this import

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
            # st.session_state.chat_log.append({"name": "assistant", "msg": input_code})
            
            for output in tool_call.code_interpreter.outputs:
                if output.type == "logs":
                    output_text = f"### code interpreter\noutput:\n```\n{output.logs}\n```"
                    with st.chat_message("Assistant"):
                        # st.markdown(output_text, True)
                        st.session_state.chat_log.append(
                            {"name": "assistant", "msg": output_text}
                        )
                elif output.type == "image":
                    # Handle image outputs
                    with st.chat_message("Assistant"):
                        st.image(f"data:image/png;base64,{output.image.file_id}")
                        
        elif tool_call.type == "function":
            # Function calls are now handled in the main run loop
            print(f"Function call detected: {tool_call.function.name}")

    # Note: on_run_requires_action is removed - function calls handled in main run loop

    def handle_function_call(self, tool_call):
        """Handle function tool calls - deprecated in favor of on_run_requires_action"""
        print(f"Function call handled via on_run_requires_action: {tool_call.function.name}")
        tool_call_id = tool_call.id
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        # Run the tool
        if function_name in TOOL_MAP:
            output = TOOL_MAP[function_name](**arguments)

        return output

    @override
    def on_exception(self, exception):
        print(f"Exception occurred: {exception}")
        st.error(f"An error occurred: {str(exception)}")


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


# def create_file_link(file_name, file_id):
#     """Create downloadable file link"""
#     try:
#         content = client.files.content(file_id)
#         content_type = content.response.headers["content-type"]
#         b64 = base64.b64encode(content.text.encode(content.encoding)).decode()
#         link_tag = f'<a href="data:{content_type};base64,{b64}" download="{file_name}">Download Link</a>'
#         return link_tag
#     except Exception as e:
#         print(f"Error creating file link: {e}")
#         return "Download Link"


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

def run_stream(user_input, selected_assistant_id):
    """Run the assistant with streaming"""
    try:
        want_run = True

        if want_run:
            # Create thread if it doesn't exist
            if "thread" not in st.session_state or st.session_state.thread is None:
                st.session_state.thread = create_thread(user_input)
                if st.session_state.thread is None:
                    return
            
            # Create message
            message = create_message(st.session_state.thread, user_input)
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
            
            # Start run (non-streaming for function calls)
            print(f"Starting run with assistant: {selected_assistant_id}")
            run = client.beta.threads.runs.create(
                thread_id=st.session_state.thread.id,
                assistant_id=selected_assistant_id,
            )
            
            # Poll for completion and handle function calls
            max_iterations = 60  # Increased timeout for complex processing
            iteration = 0
            
            # Create loading animation placeholder outside chat message context
            loading_placeholder = st.empty()
                
            # Loading messages for variety
            loading_messages = [
                "🤖 Analyzing your request",
                "🔍 Searching legal documents", 
                "📋 Processing information",
                "⚖️ Reviewing legal requirements",
                "📝 Preparing response"
            ]
                
            while iteration < max_iterations:
                iteration += 1
                
                # Show loading animation with cycling dots and messages
                loading_dots = "." * ((iteration % 3) + 1)
                message_index = (iteration // 3) % len(loading_messages)
                current_message = loading_messages[message_index]
                loading_placeholder.markdown(f"{current_message}{loading_dots}")
                
                # Get current run status
                run = client.beta.threads.runs.retrieve(
                    thread_id=st.session_state.thread.id,
                    run_id=run.id
                )
                print(f"Run status: {run.status} (iteration {iteration})")
                
                if run.status == "completed":
                    print("✅ Run completed successfully!")
                    loading_placeholder.empty()  # Clear loading animation
                    break
                elif run.status == "requires_action":
                    print("🔧 Function calling required!")
                    loading_placeholder.markdown("🔧 Executing tools...")
                    
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls
                    tool_outputs = []
                    
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        # Update loading message for specific function
                        if function_name == "read_file":
                            loading_placeholder.markdown("📖 Reading document...")
                        elif function_name == "write_file":
                            loading_placeholder.markdown("💾 Creating document...")
                        elif function_name == "search_documents":
                            loading_placeholder.markdown("🔍 Searching legal database...")
                        else:
                            loading_placeholder.markdown(f"⚙️ Executing {function_name}...")
                        
                        print(f"  Calling: {function_name}({function_args})")
                        
                        if function_name in TOOL_MAP:
                            try:
                                output = TOOL_MAP[function_name](**function_args)
                                print(f"  Result: {output[:100]}..." if len(str(output)) > 100 else f"  Result: {output}")
                                
                                tool_outputs.append({
                                    "tool_call_id": tool_call.id,
                                    "output": str(output)
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
                    loading_placeholder.empty()  # Clear loading animation
                    if run.last_error:
                        print(f"Error: {run.last_error}")
                    break
                elif run.status in ["queued", "in_progress"]:
                    # Continue polling
                    import time
                    time.sleep(1)
                else:
                    print(f"Unknown status: {run.status}")
                    break
            
            if iteration >= max_iterations:
                print("❌ Run timed out")
                loading_placeholder.empty()  # Clear loading animation
                return
                
            # Get the final messages and display them
            loading_placeholder.markdown("✅ Preparing response...")
            time.sleep(0.5)  # Brief pause before showing response
            loading_placeholder.empty()  # Clear loading animation
            
            messages = client.beta.threads.messages.list(thread_id=st.session_state.thread.id)
            
            # Store download info in session state to persist across reruns
            download_info = None
            
            for message in messages.data:
                if message.role == "assistant" and message.created_at > (time.time() - 300):  # Recent messages
                    for content in message.content:
                        if content.type == "text":
                            formatted_text = format_annotation(content.text)
                            
                            # Replace URLs in the text for display
                            display_text = formatted_text.replace('https://example.com/', os.environ.get("FILE_SERVER_BASE_URL", "http://localhost:8501/templates/"))
                            
                            # Check for download URLs
                            url_pattern = r"http?://[^\s]+?(?:%[0-9A-Fa-f]{2}|[^\s%])*\.docx(?:\?[^\s]*)?"
                            urls = re.findall(url_pattern, display_text)
                            
                            if urls:
                                print('url found')
                                url = urls[0]
                                print(f"Processing URL: {url}")
                                filename = url.split("/")[-1]
                                print(f"Extracted filename: {filename}")
                                
                                # URL decode the filename
                                filename = urllib.parse.unquote(filename)
                                print(f"Decoded filename: {filename}")
                                
                                file_path = f'templates/{filename}'
                                print(f"Looking for file at: {file_path}")
                                print(f"File exists: {os.path.exists(file_path)}")
                                
                                if os.path.exists(file_path):
                                    # Store download info for later use
                                    download_info = {
                                        'file_path': file_path,
                                        'filename': filename,
                                        'url': url
                                    }
                            
                            # Only append assistant message to chat_log, do NOT render inline
                            st.session_state.chat_log.append({"name": "assistant", "msg": display_text})
                            # Add download button info if file exists
                            if download_info:
                                st.session_state.download_info = download_info
                    break  # Only show the latest assistant message
        else:
            # with st.chat_message("Assistant"):
            #     st.markdown("Hi hello test message http://localhost:8501/templates/atto di costituzione societa tipo SNC.docx", True)

            st.session_state.chat_log.append({"name": "assistant", "msg": 'Hi hello test message http://localhost:8501/templates/atto di costituzione societa tipo SNC.docx'})
            download_info = {
                'file_path': 'templates/atto di costituzione societa tipo SNC.docx',
                'filename': 'atto di costituzione societa tipo SNC.docx',
                'url': 'http://localhost:8501/templates/atto di costituzione societa tipo SNC.docx'
            }
            # Add download button immediately after the message if file exists
            if download_info:
                st.session_state.download_info = download_info

    except Exception as e:
        print(f"Error in run_stream: {e}")
        # Clear loading animation if it exists
        if 'loading_placeholder' in locals():
            loading_placeholder.empty()
        st.error(f"Error running assistant: {str(e)}")
    finally:
        # Re-enable chat input after response
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
    chats = st.session_state.chat_log
    
    # If we just submitted a message, skip rendering the last user message to avoid duplication
    if (st.session_state.get("just_submitted", False) and 
        chats and 
        chats[-1]["name"] == "user"):
        chats = chats[:-1]  # Skip the last message if it's the user message we just showed
    
    for chat in chats:
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

if "function_calls_count" not in st.session_state:
    st.session_state.function_calls_count = 0

if "last_function_call" not in st.session_state:
    st.session_state.last_function_call = None

if "just_submitted" not in st.session_state:
    st.session_state.just_submitted = False


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
    st.session_state.function_calls_count = 0
    st.session_state.last_function_call = None


def load_chat_screen(assistant_id, assistant_title):
    """Load the main chat interface"""

    # Reset chat button
    if st.sidebar.button("Reset Chat"):
        reset_chat()
        st.session_state.download_info = None
        st.rerun()

    st.title(assistant_title if assistant_title else "Italian Law Assistant")

    user_msg = st.chat_input(
        "Ask me about legal documents...", 
        on_submit=disable_form, 
        disabled=st.session_state.in_progress
    )
    
    if user_msg:
        # Show user message immediately for better UX
        with st.chat_message("user"):
            st.markdown(user_msg, True)
        # Add to chat log and mark as just submitted to avoid duplication
        st.session_state.chat_log.append({"name": "user", "msg": user_msg})
        st.session_state.just_submitted = True
        run_stream(user_msg, assistant_id)
    else:
        st.session_state.just_submitted = False

    render_chat()

    # Show download button if available, but do NOT re-render chat
    if "download_info" in st.session_state:
        info = st.session_state.download_info
        if info:
            try:
                with open(info['file_path'], "rb") as f:
                    file_bytes = f.read()

                # Force it into its own container so it’s always visible
                with st.container():
                    st.markdown("## 📥 Your document is ready:")
                    st.download_button(
                        label="📄 Download Template",
                        data=file_bytes,
                        file_name=info['filename'],
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key="persistent_download"
                    )
            except Exception as e:
                st.error(f"Error creating download button: {e}")

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