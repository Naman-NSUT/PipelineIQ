import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os

app = FastAPI(title="PipelineIQ Demo App")

class TaskCreate(BaseModel):
    title: str

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    completed: Optional[bool] = None

class TaskResponse(BaseModel):
    id: int
    title: str
    completed: bool

# In-memory database
db: List[dict] = []
current_id = 1

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open(os.path.join(os.path.dirname(__file__), "index.html"), "r") as f:
        return f.read()

@app.get("/api/tasks", response_model=List[TaskResponse])
def get_tasks():
    return db

@app.post("/api/tasks", response_model=TaskResponse)
def add_task(task: TaskCreate):
    global current_id
    if not task.title.strip():
        raise HTTPException(status_code=400, detail="Task title cannot be empty")
    new_task = {
        "id": current_id,
        "title": task.title.strip(),
        "completed": False
    }
    db.append(new_task)
    current_id += 1
    return new_task

@app.put("/api/tasks/{task_id}", response_model=TaskResponse)
def update_task(task_id: int, task_update: TaskUpdate):
    for task in db:
        if task["id"] == task_id:
            if task_update.title is not None:
                if not task_update.title.strip():
                    raise HTTPException(status_code=400, detail="Task title cannot be empty")
                task["title"] = task_update.title.strip()
            if task_update.completed is not None:
                task["completed"] = task_update.completed
            return task
    raise HTTPException(status_code=404, detail="Task not found")

@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: int):
    global db
    for i, task in enumerate(db):
        if task["id"] == task_id:
            db.pop(i)
            return {"detail": "Task deleted"}
    raise HTTPException(status_code=404, detail="Task not found")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8555, reload=True)
