# BookDB API

A FastAPI-based REST API for BookDB

## Prerequisites

- Python 3.10+ (recommended 3.11)
- pip or [uv](https://docs.astral.sh/uv/)

## Local Development Setup

### Navigate to the API directory

`cd apps/api`

### Create a virtual environment

`bash py -m venv .venv `

### Activate the virtual environment

Windows PowerShell

`.venv\Scripts\Activate`

Windows Command Prompt

`.venv\Scripts\activate.bat`

macOS/Linux

`source .venv/bin/activate`

After activation, you should see (.venv) in your terminal.

### Install dependencies

`pip install -r requirements.txt`

If requirements.txt does not exist:

`pip install fastapi uvicorn scalar_fastapi`

### Running the API

Start the development server:

`uvicorn main:app --reload --port 8080`

The API will be available at: `http://localhost:8080`

Interactive API documentation is available at: `http://localhost:8080/docs`

## Configuration

### CORS Settings

The API is configured to accept requests from:

- `http://localhost`
- `http://localhost:8080`

To modify allowed origins, edit the `origins` list in `main.py`:

```python
origins = [
    "http://localhost",
    "http://localhost:8080",
    # Add more origins as needed
]
```

### Server Settings

- **Host**: 0.0.0.0 (accessible from any network interface)
- **Port**: 8080
- **Reload**: Enabled (auto-restarts on code changes in development)

## API Endpoints

### GET `/`

Returns a simple welcome message.

**Response:**

```json
{
  "Hello": "World"
}
```

### POST `/testfunc`

Test endpoint that increments a number and converts text to uppercase.

**Request Body:**

```json
{
  "num": 5,
  "txt": "hello"
}
```

**Response:**

```json
{
  "num": 6,
  "txt": "HELLO"
}
```

## Project Structure

```
apps/api/
├── main.py           # FastAPI application setup and entry point
├── routes.py         # API route definitions
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Dependencies

- **fastapi[standard]** (0.129.0): Web framework for building APIs
- **scalar_fastapi** (1.6.2): Interactive API documentation UI

## Development

### Code Style

The project uses standard Python conventions. Consider installing linters:

```bash
uv pip install black flake8
```

### Testing

To add tests, create a `tests/` directory with test files and run:

```bash
pytest
```

## Troubleshooting

### Port Already in Use

If port 8080 is already in use, modify the port in `main.py`:

```python
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=YOUR_PORT, reload=True)
```

### CORS Issues

If you're getting CORS errors, ensure your client's origin is added to the `origins` list in `main.py`.

## Common Tasks

### Connect to Frontend

Update `origins` in `main.py` to include your frontend URL:

```python
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",  # Vite dev server
    "http://yourfrontend.com",  # Production frontend
]
```

### Add New Endpoints

Add new routes to `routes.py`:

```python
@router.get("/new-endpoint", tags=[tag])
async def new_endpoint():
    return {"message": "New endpoint"}
```

## License

See the main repository LICENSE file.
