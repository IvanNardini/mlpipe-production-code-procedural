from procedural_ml_pipe import preprocess
import pytest

# Read configuration
stream = open('config.yaml', 'r')
config = yaml.load(stream)

def test_loader():
    result = loader(config['data_ingestion']['data_path'])
    assert result == True
    assert isinstance(result, pd.DataFrame) == True

