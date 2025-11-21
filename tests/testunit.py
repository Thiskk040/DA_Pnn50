import sys
import os
import polars as pl
import tempfile
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from FindPressV2 import generate_file_paths, process_vr_data_press_and_nearest_answer

def test_generate_file_paths_basic():
    base_path  = "TestFolder"
    participants = 3
    avatars = ["Man","Robot"]
    
    expected_keys = [
        "0000_Man", "0000_Robot",
        "0001_Man", "0001_Robot",
        "0002_Man", "0002_Robot"
    ]
    expected_paths = {
        key: os.path.join(base_path, f"{key}.csv") for key in expected_keys
    }
    
    result = generate_file_paths(base_path=base_path,avatars=avatars,participants=participants)
    assert len(result) == participants*len(avatars)
    assert result == expected_paths

def test_generate_file_paths_default_args():
    base_path = "Data"
    result = generate_file_paths(base_path)
    
    assert len(result) ==  31*3 #31 คน 3 Avatars
    assert "0000_Man" in result
    assert result["0030_Woman"] == os.path.join(base_path,"0030_Woman.csv")
    
#test function 2 ---->nearest ans    
def create_mock_csv(file_path, data: pl.DataFrame):
    data.write_csv(file_path)

@pytest.fixture
def mock_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        df1 = pl.DataFrame({
            "User_Action": ["press_button", None, "Answer_question"],
            "time_from_start": [1.0, 2.0, 1.1],
            "Form_Question": [None, None, "Q1"],
            "Form_Answer": [None, None, "A1"]
        })
        df2 = pl.DataFrame({
            "User_Action": ["walk", "press_key", "Answer_something"],
            "time_from_start": [0.5, 2.5, 2.6],
            "Form_Question": [None, None, "Q2"],
            "Form_Answer": [None, None, "A2"]
        })

        f1 = os.path.join(tmpdir, "0000_Man.csv")
        f2 = os.path.join(tmpdir, "0001_Woman.csv")
        create_mock_csv(f1, df1)
        create_mock_csv(f2, df2)

        files = {
            "0000_Man": f1,
            "0001_Woman": f2
        }
        yield files  # return dict to test

def test_process_vr_data_press_and_nearest_answer(mock_files):
    df_result, output_path = process_vr_data_press_and_nearest_answer(mock_files)
    assert os.path.exists(output_path)
    df_out = pl.read_csv(output_path)
    expected_cols = ["User_Action", "time_from_start", "Form_Question", "Form_Answer", "Source_File"]
    for col in expected_cols:
        assert col in df_out.columns
    assert df_out.height > 0
    unique_sources = df_out["Source_File"].unique().to_list()
    for source in unique_sources:
        assert source in mock_files.keys()
