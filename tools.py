import os
import openai
import json
from pathlib import Path

openai_api_key = os.environ.get("OPENAI_API_KEY")


def _templates_dir() -> Path:
    """Return the templates directory (env TEMPLATES_DIR or ./templates)."""
    env_dir = os.environ.get("TEMPLATES_DIR")
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
        print(f"Using TEMPLATES_DIR from environment: {path}")
        return path
    path = Path(__file__).parent.joinpath("templates").resolve()
    print(f"Using default templates directory: {path}")
    return path

def list_available_files():
    """
    Return the list of available files from local templates folder only.
    """
    try:
        tdir = _templates_dir()
        print(f"Checking templates directory: {tdir}")
        
        if tdir.exists() and tdir.is_dir():
            print(f"Found local templates directory with {len(list(tdir.iterdir()))} items")
            entries = []
            for p in sorted(tdir.iterdir()):
                if p.is_file() and not p.name.startswith('.'):
                    try:
                        size = p.stat().st_size
                    except OSError:
                        size = None
                    entries.append({
                        "filename": p.name,
                        "format": 'txt',
                        "size": size,
                        "source": "local",
                    })
            print(f"Returning {len(entries)} local files")
            return json.dumps(entries, ensure_ascii=False)
        else:
            print("Templates directory does not exist")
            return json.dumps({"error": "Templates directory not found", "files": []})

    except Exception as e:
        err = f"Error listing files: {e}"
        print(err)
        return json.dumps({"error": err, "files": []})

def get_file_content(file_name: str, format: str = "text"):
    """
    Retrieve file content by name from local templates only.
    - file_name: name of the file, e.g. 'deed_snc.docx'
    - format: 'text' or 'docx'
    """
    try:
        print(f"tools.get_file_content invoked name={file_name} format={format}")
        tdir = _templates_dir()

        # Check if templates directory exists
        if not tdir.exists() or not tdir.is_dir():
            error_msg = f"Templates directory '{tdir}' does not exist"
            print(error_msg)
            return error_msg

        # Check if file exists
        target = tdir.joinpath(file_name)
        if not target.exists() or not target.is_file():
            error_msg = f"File '{file_name}' not found in templates folder."
            print(error_msg)
            return error_msg

        # Return content based on format
        if format == "text":
            try:
                content = target.read_text(encoding="utf-8")
                print(f"Successfully read text content ({len(content)} characters)")
                return content
            except UnicodeDecodeError:
                content = target.read_text(encoding="latin-1", errors="ignore")
                print(f"Successfully read text content with latin-1 encoding ({len(content)} characters)")
                return content
        elif format == "docx":
            result = json.dumps({"download_local": {"path": str(target), "filename": file_name}})
            print(f"Returning docx download info for {file_name}")
            return result
        else:
            error_msg = f"Unsupported format '{format}'. Use 'text' or 'docx'."
            print(error_msg)
            return error_msg

    except Exception as e:
        err = f"Error retrieving file '{file_name}': {e}"
        print(err)
        return err

def get_file_link(file_name: str, base_url: str = None):
    """
    Return the download/access link for a specific file.
    - file_name: name of the file, e.g. 'atto di costituzione societa tipo SRL.txt'
    - base_url: optional base URL for the file server (from env FILE_SERVER_BASE_URL or default)
    """
    try:
        print(f"tools.get_file_link invoked name={file_name}")
        tdir = _templates_dir()

        # Check if templates directory exists
        if not tdir.exists() or not tdir.is_dir():
            error_msg = f"Templates directory '{tdir}' does not exist"
            print(error_msg)
            return json.dumps({"error": error_msg})

        # Check if file exists
        target = tdir.joinpath(file_name)
        if not target.exists() or not target.is_file():
            error_msg = f"File '{file_name}' not found in templates folder."
            print(error_msg)
            return json.dumps({"error": error_msg})

        # Get base URL from environment or use default
        if not base_url:
            base_url = os.environ.get("FILE_SERVER_BASE_URL", "http://localhost:8501/templates")
        
        # Create the file link
        file_link = f"{base_url.rstrip('/')}/{file_name}"
        
        # Get file stats
        try:
            file_stats = target.stat()
            file_size = file_stats.st_size
            file_modified = file_stats.st_mtime
        except OSError:
            file_size = None
            file_modified = None

        result = {
            "filename": file_name,
            "link": file_link,
            "local_path": str(target),
            "size": file_size,
            "last_modified": file_modified,
            "status": "available"
        }
        
        print(f"Successfully generated link for {file_name}: {file_link}")
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        err = f"Error generating link for file '{file_name}': {e}"
        print(err)
        return json.dumps({"error": err})

TOOL_MAP = {
    "list_available_files": list_available_files,
    "get_file_content": get_file_content,
    "get_file_link": get_file_link,
}