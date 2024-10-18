from chatbot_model_training.chatbot_training import train_chatbot_model
import unittest
import os
import shutil
import pickle
import tensorflow as tf
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env
load_dotenv()

class TestChatbotModelTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Paths for the files that will be created during training
        cls.source_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'chatbot_model_training')
        cls.model_path = os.path.join(cls.source_dir, 'chatbot_model.keras')
        cls.words_path = os.path.join(cls.source_dir, 'chatbot_words.pkl')
        cls.classes_path = os.path.join(cls.source_dir, 'chatbot_classes.pkl')
        cls.intents_file = os.path.join(cls.source_dir, 'chatbot_intents.json')

        # Get the archive directory from the environment variable
        cls.archive_dir = os.getenv('PROJECT_ARCHIVE_DIRECTORY')

        # Ensure the archive directory exists
        if not os.path.exists(cls.archive_dir):
            os.makedirs(cls.archive_dir)

        # Check that the training data (chatbot_intents.json) exists before proceeding
        cls.assert_training_data_exists()
        
        # Move existing files to archive instead of deleting
        cls.move_to_archive(cls.model_path)
        cls.move_to_archive(cls.words_path)
        cls.move_to_archive(cls.classes_path)

        # Run the training function once for all tests
        train_chatbot_model()

    @classmethod
    def assert_training_data_exists(cls):
        """Check if chatbot_intents.json exists before running the training."""
        if not os.path.exists(cls.intents_file):
            raise FileNotFoundError(f"Required training data file not found: {cls.intents_file}")
        else:
            print(f"Training data found: {cls.intents_file}")
            
    @classmethod
    def move_to_archive(cls, file_path):
        if os.path.exists(file_path):
            # Generate a timestamp and append to the filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = os.path.basename(file_path)
            new_name = f"{os.path.splitext(base_name)[0]}_{timestamp}{os.path.splitext(base_name)[1]}"
            archive_path = os.path.join(cls.archive_dir, new_name)

            # Move the file to the archive with the new name
            shutil.move(file_path, archive_path)

    def test_model_training_creates_files(self):
        # Test if the model file is created
        self.assertTrue(os.path.exists(self.model_path), "Model file was not created.")
        # Test if words and classes pickle files are created
        self.assertTrue(os.path.exists(self.words_path), "Words pickle file was not created.")
        self.assertTrue(os.path.exists(self.classes_path), "Classes pickle file was not created.")

    def test_model_architecture(self):
        # Print the model path for debugging
        print(f"Trying to load model from: {self.model_path}")

        try:
            model = tf.keras.models.load_model(self.model_path)
            
            # Test if model has the expected number of layers
            self.assertEqual(len(model.layers), 5, "Model should have 5 layers.")
            # Test if the first layer has 128 units as expected
            self.assertEqual(model.layers[0].units, 128, "First layer should have 128 units.")
            # Test if the last layer has softmax activation
            self.assertEqual(model.layers[-1].activation.__name__, 'softmax', "Last layer should have softmax activation.")
        
        except ValueError as e:
            self.fail(f"Failed to load the model due to: {str(e)}")
        
    def test_pickle_file_content(self):
        # Load the pickle files
        with open(self.words_path, 'rb') as f:
            words = pickle.load(f)
        with open(self.classes_path, 'rb') as f:
            classes = pickle.load(f)
        
        # Test if the words list is not empty
        self.assertTrue(len(words) > 0, "Words list is empty.")
        # Test if the classes list is not empty
        self.assertTrue(len(classes) > 0, "Classes list is empty.")
        # Optionally, check if specific words or classes exist

if __name__ == '__main__':
    unittest.main()
