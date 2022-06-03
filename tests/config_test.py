from source.config import Config


def test_config():
    testConfig = Config("tests/model_test.yaml")
    assert testConfig.value("model.name") == "testModel"
