import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging
from contextlib import contextmanager

from .conversation_cleanup import ConversationCleanup
from .db_ops import get_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global task registry
background_tasks: Dict[str, asyncio.Task] = {}

async def run_periodic_task(
    task_func,
    interval_minutes: int,
    task_name: str,
    *args,
    **kwargs
) -> None:
    """Run a task periodically at specified intervals."""
    while True:
        try:
            logger.info(f"Running {task_name}")
            # Get a new database session for each run
            db = next(get_db())
            try:
                cleanup = ConversationCleanup(db)
                cleanup.mark_inactive_conversations()
                cleanup.archive_old_conversations()
                logger.info(f"Completed {task_name}")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error in {task_name}: {str(e)}")
        
        await asyncio.sleep(interval_minutes * 60)

async def start_background_tasks() -> None:
    """Initialize and start background tasks."""
    try:
        # Start conversation cleanup task
        cleanup_task = asyncio.create_task(
            run_periodic_task(
                task_func=None,  # Not used anymore since we handle it directly
                interval_minutes=15,
                task_name="conversation_cleanup"
            )
        )
        background_tasks["cleanup"] = cleanup_task
        
        logger.info("Background tasks started successfully")
    except Exception as e:
        logger.error(f"Error starting background tasks: {str(e)}")
        raise

def stop_background_tasks() -> None:
    """Stop all running background tasks."""
    for task_name, task in background_tasks.items():
        if not task.done():
            task.cancel()
            logger.info(f"Cancelled {task_name} task") 