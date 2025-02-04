import os 
import unittest
from unittest.mock import patch, MagicMock


from ..src.file_mangement.directory_creator import create_directory


class  TestCreatDirectory(unittest.TestCase):
    
    @patch("..src.file_mangement.directory_creator.os.makedirs")
    @patch("..src.file_mangement.directory_creator.os.path.exists")
    def test_create_directory_new_directory(sefl, mock_exists, mock_makedirs):

        directory_path = "/i/hate/dust_rectory"

        mock_exists.return_value = False # this simulates that the directory does not  exist

        #trigger the function
        create_directory(directory_path)

        #checks that the directory path was used with the os.path.exists code 
        mock_exists.assert_called_once_with(directory_path)

        # checks that the make os.makedirs method was called once with the directory path
        mock_makedirs.assert_called_once_with(directory_path)

    @patch('..src.file_mangement.directory_creator.logging.error')
    @patch('..src.file_mangement.directory_creator.os.path.exists')
    def test_create_directory_existing_directory(self, mock_exists, mock_error):
        # Arrange
        directory_path = '/path/to/existing/directory'
        mock_exists.return_value = True  # Simulate that the directory already exists

        # Act
        create_directory(directory_path)

        # Assert
        mock_exists.assert_called_once_with(directory_path)  # Check if os.path.exists was called with the correct path
        mock_error.assert_called_once_with(f"Directory '{directory_path}' already exists.")  # Check if logging.error was called with the correct message




if __name__ == '__main__':
    unittest.main()