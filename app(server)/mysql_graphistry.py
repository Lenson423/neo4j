from typing import List, Dict, Any, Type, Optional
from pydantic import BaseModel, Extra, ValidationError
from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session as SQLASession


class ModularityParams(BaseModel):
    iterations: int
    min_size: int
    class Config:
        extra = Extra.ignore

class LouvainParams(BaseModel):
    iterations: int
    min_size: int
    class Config:
        extra = Extra.ignore

class KClusteringParams(BaseModel):
    clusters: int
    iterations: int
    class Config:
        extra = Extra.ignore

class SBMParams(BaseModel):
    clusters: int
    class Config:
        extra = Extra.ignore

class VisualizationRequest(BaseModel):
    algorithm: str
    algorithm_params: Dict
    edge_type: List[str]
    metric: str

class ResultRequest(BaseModel):
    algorithm: str
    algorithm_params: Dict
    edge_type: List[str]
    metric: str
    url: str

Base = declarative_base()

class VisualizationRequestDB(Base):
    __tablename__ = "graphistry_result"

    id = Column(Integer, primary_key=True)
    algorithm = Column(String, nullable=False)
    algorithm_params = Column(JSON, nullable=False)
    edge_type = Column(JSON, nullable=False)
    metric = Column(String, nullable=False)
    url = Column(String, nullable=False)

class VisualizationRequestManager:
    def __init__(self, db_url: str = "sqlite:///graphistry_result.db"):
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        self.param_schemas: Dict[str, Type[BaseModel]] = {
            "modularity": ModularityParams,
            "louvain": LouvainParams,
            "kclustering": KClusteringParams,
            "sbm": SBMParams
        }

    def save(self, data: Dict[str, Any]) -> int:
        session: SQLASession = self.Session()
        try:
            request = ResultRequest(**data)

            if request.algorithm not in self.param_schemas:
                raise ValueError(f"Unsupported algorithm: {request.algorithm}")

            schema_cls = self.param_schemas[request.algorithm]
            try:
                schema_cls(**request.algorithm_params)
            except ValidationError as e:
                raise ValueError(f"Invalid parameters for '{request.algorithm}': {e}")

            record = VisualizationRequestDB(**request.dict())
            session.add(record)
            session.commit()
            return record.id

        finally:
            session.close()

    def get_by_params(
        self,
        algorithm: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Ищет записи по заданному algorithm, одновременно учитывая только те поля
        params, которые имеются в Pydantic-схеме для этого algorithm. Все остальные ключи
        во входном params игнорируются.

        :param algorithm: название алгоритма ("modularity", "louvain", "kclustering" или "sbm").
        :param params: словарь с теми полями, по которым нужно искать (например, {"iterations": 10, "min_size": 5}).
                       Если params = None или пуст, вернётся вообще все записи по этому algorithm.
        :return: список словарей со всеми полями найденных записей (id, algorithm, algorithm_params, edge_type, metric, url).
        """
        session: SQLASession = self.Session()
        try:
            if algorithm is None:
                all_recs = session.query(VisualizationRequestDB).all()
                return [self._to_dict(r) for r in all_recs]

            if algorithm not in self.param_schemas:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            if not params:
                all_recs = session.query(VisualizationRequestDB).filter_by(algorithm=algorithm).all()
                return [self._to_dict(r) for r in all_recs]

            schema_cls = self.param_schemas[algorithm]
            try:
                validated: BaseModel = schema_cls(**params)
            except ValidationError as e:
                raise ValueError(f"Invalid search parameters for '{algorithm}': {e}")

            search_dict: Dict[str, Any] = validated.dict()

            all_recs = session.query(VisualizationRequestDB).filter_by(algorithm=algorithm).all()

            def matches(r_params: Dict[str, Any], to_find: Dict[str, Any]) -> bool:
                for key, val in to_find.items():
                    if str(r_params.get(key)) != str(val):
                        return False
                return True

            matched: List[VisualizationRequestDB] = [
                rec for rec in all_recs if matches(rec.algorithm_params, search_dict)
            ]

            return [self._to_dict(r) for r in matched]

        finally:
            session.close()

    def _to_dict(self, record: VisualizationRequestDB) -> Dict[str, Any]:
        return {
            "id": record.id,
            "algorithm": record.algorithm,
            "algorithm_params": record.algorithm_params,
            "edge_type": record.edge_type,
            "metric": record.metric,
            "url": record.url,
        }

