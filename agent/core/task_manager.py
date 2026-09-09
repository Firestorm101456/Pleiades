from __future__ import annotations

import json

from agent.capabilities.models import utc_now
from agent.core.tasks import Task, TaskAttempt, TaskStatus
from agent.memory.database import MemoryDatabase


class TaskManager:
    def __init__(self, db: MemoryDatabase) -> None:
        self.db = db
        self.tasks: dict[str, Task] = {}

    def register(self, tasks: list[Task]) -> None:
        for task in tasks:
            if task.task_id not in self.tasks:
                self.tasks[task.task_id] = task
            self.persist_task(self.tasks[task.task_id])
        self.refresh_ready_tasks()

    def add_task(self, task: Task) -> None:
        self.tasks[task.task_id] = task
        self.persist_task(task)
        self.refresh_ready_tasks()

    def refresh_ready_tasks(self) -> None:
        completed = {task.task_id for task in self.tasks.values() if task.status == TaskStatus.COMPLETED}
        for task in self.tasks.values():
            if task.status in {TaskStatus.CREATED, TaskStatus.WAITING} or (task.status == TaskStatus.BLOCKED and not task.failure_information):
                task.status = TaskStatus.READY if set(task.dependencies) <= completed else TaskStatus.BLOCKED
                task.updated_at = utc_now()
                self.persist_task(task)

    def ready_tasks(self) -> list[Task]:
        self.refresh_ready_tasks()
        return sorted(
            [task for task in self.tasks.values() if task.status == TaskStatus.READY],
            key=lambda item: (item.deadline is not None, item.priority, -item.estimated_duration),
            reverse=True,
        )

    def blocked_tasks(self) -> list[Task]:
        self.refresh_ready_tasks()
        return [task for task in self.tasks.values() if task.status == TaskStatus.BLOCKED]

    def start(self, task_id: str) -> Task:
        return self.transition(task_id, TaskStatus.IN_PROGRESS)

    def complete(self, task_id: str, observation: str) -> Task:
        task = self.tasks[task_id]
        task.add_observation(observation)
        task.status = TaskStatus.COMPLETED
        task.updated_at = utc_now()
        self.persist_task(task)
        self.refresh_ready_tasks()
        return task

    def fail(self, task_id: str, reason: str) -> Task:
        task = self.tasks[task_id]
        task.status = TaskStatus.FAILED
        task.failure_information.append(reason)
        task.updated_at = utc_now()
        self.persist_task(task)
        return task

    def block(self, task_id: str, reason: str) -> Task:
        task = self.tasks[task_id]
        task.status = TaskStatus.BLOCKED
        if reason not in task.failure_information:
            task.failure_information.append(reason)
        task.updated_at = utc_now()
        self.persist_task(task)
        return task

    def transition(self, task_id: str, status: TaskStatus) -> Task:
        task = self.tasks[task_id]
        allowed = {
            TaskStatus.CREATED: {TaskStatus.READY, TaskStatus.BLOCKED, TaskStatus.CANCELLED},
            TaskStatus.READY: {TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, TaskStatus.CANCELLED},
            TaskStatus.IN_PROGRESS: {TaskStatus.COMPLETED, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.WAITING},
            TaskStatus.BLOCKED: {TaskStatus.READY, TaskStatus.FAILED, TaskStatus.CANCELLED},
            TaskStatus.WAITING: {TaskStatus.READY, TaskStatus.BLOCKED, TaskStatus.CANCELLED},
            TaskStatus.COMPLETED: set(),
            TaskStatus.FAILED: set(),
            TaskStatus.CANCELLED: set(),
        }
        if status not in allowed[task.status]:
            raise ValueError(f"Invalid task transition: {task.status.value} -> {status.value}")
        task.status = status
        task.updated_at = utc_now()
        self.persist_task(task)
        return task

    def record_attempt(self, task_id: str, action: str, status: str, observation: str) -> None:
        task = self.tasks[task_id]
        task.attempts.append(TaskAttempt(utc_now(), action, status, observation))
        task.updated_at = utc_now()
        self.persist_task(task)

    def objective_complete(self, objective_id: str) -> bool:
        objective_tasks = [task for task in self.tasks.values() if task.objective_id == objective_id]
        return bool(objective_tasks) and all(task.status == TaskStatus.COMPLETED for task in objective_tasks)

    def persist_task(self, task: Task) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tasks
                (task_id, objective_id, parent_task_id, status, title, priority, payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.objective_id,
                    task.parent_task_id,
                    task.status.value,
                    task.title,
                    task.priority,
                    self.db.dumps(task.to_record()),
                    task.updated_at,
                ),
            )

    def load_objective(self, objective_id: str) -> list[Task]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT payload FROM tasks WHERE objective_id = ?", (objective_id,)).fetchall()
        tasks = [Task.from_record(json.loads(row["payload"])) for row in rows]
        self.tasks.update({task.task_id: task for task in tasks})
        return tasks
