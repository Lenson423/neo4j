import asyncio
import hashlib
import json
import os

from celery.exceptions import MaxRetriesExceededError
from fastapi import FastAPI, Request, HTTPException, APIRouter
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Dict
from celery import Celery
from fastapi.middleware.cors import CORSMiddleware

from mysql_graphistry import VisualizationRequest, VisualizationRequestManager
from sbm import sbm
from neo4j_dao import Neo4jDAO
from visualiser import visualise_graphistry
from graph_analyzer import ensure_graph_exists, GraphAnalyzer

app = FastAPI(title="Graph Visualization API with Celery", version="1.0.0")
router = APIRouter(prefix="/zvcolztkcjhdwohlbsgutww")

celery_app = Celery(
    'worker',
    broker='redis://localhost:6379',
    backend='redis://localhost:6379/'
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = VisualizationRequestManager()

@celery_app.task(bind=True, default_retry_delay=30 * 60, max_retries=10)
def run_graph_analysis(self, request_data: Dict, task_id: str):
    try:

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        total_steps = 3
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': total_steps,
                'status': f'Processing step {0}/{total_steps}: creating graph'
            }
        )
        ga = GraphAnalyzer()
        session = loop.run_until_complete(ga.get_session())
        loop.run_until_complete(ensure_graph_exists(session, task_id))
        self.update_state(
            state='PROGRESS',
            meta = {
                        'current': 1,
                        'total': total_steps,
                        'status': f'Processing step {1}/{total_steps}: running algorithm (it may took a lot of time)'
                    }
        )

        if request_data["algorithm"] == "modularity":
            iterations, min_size = (int(request_data["algorithm_params"]["iterations"]),
                                               int(request_data["algorithm_params"]["min_size"]))
            data = loop.run_until_complete(ga.analyze(session, task_id, iterations, 1e-4, min_size))
        elif request_data["algorithm"] == "louvain":
            iterations, min_size = (int(request_data["algorithm_params"]["iterations"]),
                                               int(request_data["algorithm_params"]["min_size"]))
            neo4j = Neo4jDAO()
            data = loop.run_until_complete(neo4j.louvain(session, task_id, iterations, 1e-4, min_size))
        elif request_data["algorithm"] == "kclustering":
            k, iterations = (int(request_data["algorithm_params"]["clusters"]),
                                               int(request_data["algorithm_params"]["iterations"]))
            neo4j = Neo4jDAO()
            data = loop.run_until_complete(neo4j.maxkcut(session, task_id, k, iterations))
        elif request_data["algorithm"] == "sbm":
            k = int(request_data["algorithm_params"]["clusters"])
            data = sbm(k)
        else:
            loop.close()
            raise HTTPException(status_code=404, detail="Algorithm not found")

        self.update_state(
            state='PROGRESS',
            meta={
                'current': 2,
                'total': total_steps,
                'status': f'Processing step {2}/{total_steps}: visualising graph'
            }
        )

        graphistry_url, df = visualise_graphistry(request_data["metric"], data)

        tmp = {k: v for k, v in request_data.items()}
        tmp["url"] = graphistry_url
        manager.save(tmp)

        os.makedirs("static/graph/my", exist_ok=True)
        df.to_json(f"static/graph/my/{task_id}.json", orient='records', indent=2)

        loop.close()
        return {
            'status': 'COMPLETED',
            'result': {
                'url': graphistry_url,
                'request_data': request_data,
                'task_id': task_id
            }
        }
    except Exception as exc:
        try:
            raise self.retry(exc=exc)
        except MaxRetriesExceededError:
            return {
                'status': 'FAILED',
                'error': f'Max retries exceeded: {str(exc)}'
            }


@router.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Graph Visualization Tool"
    })


@router.get("/table/{filename}")
async def table(filename:str, request: Request):
    return templates.TemplateResponse("table.html", {
        "request": request,
        "title": "Graph Visualization Tool",
        "file_name": filename,
    })

@router.post("/visualize")
def visualize(request: VisualizationRequest):
    serialized_data = json.dumps(request.dict(), sort_keys=True)
    task_id = hashlib.sha256(serialized_data.encode()).hexdigest()[:20]

    existing_task = celery_app.AsyncResult(task_id)
    if existing_task.state == 'SUCCESS':
        return {
            "message": "Visualization finished with Celery",
            "task_id": task_id,
            "status_url": f"/zvcolztkcjhdwohlbsgutww/result/{task_id}"
        }
    elif existing_task.state in ['PROGRESS']:
        return {
            "message": "Task is already being processed",
            "task_id": task_id,
            "status_url": f"/zvcolztkcjhdwohlbsgutww/result/{task_id}"
        }

    data = manager.get_by_params(request.dict()["algorithm"], request.dict()["algorithm_params"])
    if data is not None and len(data) > 0:
        return {
            'status': 'COMPLETED',
            'result': {
                'url': data[0]["url"],
                'request_data': request.dict(),
                'task_id': task_id
            }
        }

    task = run_graph_analysis.apply_async(
        args=[request.dict(), task_id],
        task_id=task_id
    )

    return {
        "message": "Visualization started with Celery",
        "task_id": task_id,
        "status_url": f"/zvcolztkcjhdwohlbsgutww/result/{task_id}"
    }


@router.get("/result/{task_id}")
async def get_result(task_id: str, request: Request):
    task_result = celery_app.AsyncResult(task_id)

    if task_result.state == 'PENDING':
        response = {
            "request": request,
            'status': 'PENDING',
            'message': 'Task is pending or not found'
        }
    elif task_result.state == 'PROGRESS':
        response = {
            "request": request,
            'status': 'PROGRESS',
            'progress': task_result.info.get('current', 0),
            'total': task_result.info.get('total', 1),
            'message': task_result.info.get('status', '')
        }
    elif task_result.state == 'SUCCESS':
        if task_result.info['status'] == 'COMPLETED':
            return templates.TemplateResponse(
                "result.html",
                {
                    "request": request,
                    "task_id": task_id,
                    "graphistry_url": task_result.info['result']['url'],
                }
            )

        else:
            response = {
                "request": request,
                'status': 'FAILED',
                'error': task_result.info.get('error', 'Unknown error')
            }
    else:
        response = {
            "request": request,
            'status': 'FAILED',
            'error': str(task_result.info)
        }

    return templates.TemplateResponse("status.html", response)


@router.get("/tasks")
async def list_tasks():
    inspector = celery_app.control.inspect()
    return {
        "active": inspector.active(),
        "scheduled": inspector.scheduled(),
        "reserved": inspector.reserved()
    }

@router.get("/all_results")
async def list_tasks():
    return {
        "results": manager.get_by_params()
    }

app.include_router(router)
