# worker.py
import os
import redis
from rq import Worker, Queue
import logging

from common_app_funcs import get_redis_conn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

redis_conn = get_redis_conn()

# Start worker
if __name__ == '__main__':
    # Define which queues this worker should listen to
    queue_names = ['image_processing']
    
    # Create queues
    queues = [Queue(name, connection=redis_conn) for name in queue_names]
    
    # Create and start worker
    worker = Worker(queues)
    worker.work(with_scheduler=True)