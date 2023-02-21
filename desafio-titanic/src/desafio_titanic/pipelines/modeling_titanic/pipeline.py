"""
This is a boilerplate pipeline 'modeling_titanic'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import fill_null, feature_engineering, data_preprocessing, predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=fill_null,
            inputs=["train", "test"],
            outputs=["train_1", "test_1"],
            name="fill_null"
        ),

        node(
            func=feature_engineering,
            inputs=["train_1", "test_1"],
            outputs=["train_2", "test_2", "test_PassengerId"],
            name="feature_engineering"
        ),

        node(
            func=data_preprocessing,
            inputs=["train_2", "test_2"],
            outputs=["train_fixed", "test_fixed"],
            name="data_preprocessing"
        ),

        node(
            func=predict,
            inputs=["train_fixed", "test_fixed", "test_PassengerId"],
            outputs= "results",
            name="predict"
        )])
