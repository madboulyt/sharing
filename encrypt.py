import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from cryptography.fernet import Fernet
from typing import Union, Optional

class FileEncryption:
    def __init__(self, key: Optional[bytes] = None):
        """Initialize encryption with optional key"""
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def encrypt_file(self, filepath: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Encrypt a file and save to specified path
        
        Args:
            filepath: Path to file to encrypt
            output_path: Optional path to save encrypted file
            
        Returns:
            Path to encrypted file
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Input file not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.cipher.encrypt(data)
        encrypted_path = str(output_path or (filepath.with_suffix('.encrypted')))
        
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_data)
        
        return encrypted_path
    
    def encrypt_folder(self, folder_path: Union[str, Path], 
                      output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Encrypt an entire folder by creating a zip archive and encrypting it
        
        Args:
            folder_path: Path to folder to encrypt
            output_path: Optional path to save encrypted folder
            
        Returns:
            Path to encrypted folder archive
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Input folder not found: {folder_path}")
            
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
            temp_zip_path = temp_zip.name
        
        try:
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in folder_path.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(folder_path)
                        zipf.write(file_path, relative_path)
            
            encrypted_path = str(output_path or folder_path.with_suffix('.enc'))
            with open(temp_zip_path, 'rb') as f:
                zip_data = f.read()
            
            encrypted_data = self.cipher.encrypt(zip_data)
            
            with open(encrypted_path, 'wb') as f:
                f.write(encrypted_data)
                
            return encrypted_path
            
        finally:
            if os.path.exists(temp_zip_path):
                os.unlink(temp_zip_path)
    
    def save_key(self, key_path: Union[str, Path]) -> None:
        """Save the encryption key to file"""
        with open(key_path, 'wb') as f:
            f.write(self.key)
    
    def load_key(self, key_path: Union[str, Path]) -> None:
        """Load the encryption key from file"""
        key_path = Path(key_path)
        if not key_path.exists():
            raise FileNotFoundError(f"Key file not found: {key_path}")
            
        with open(key_path, 'rb') as f:
            self.key = f.read()
        self.cipher = Fernet(self.key)

def main():
    """Main execution function"""
    enc = FileEncryption()
    
    enc.save_key('secret.key')
    print(f"Your key: {enc.key.decode()}")
    
    base_path = Path('/home/moath/osos-ingest-serving/checkpoints')
    output_path = base_path
    
    enc.encrypt_file(
        base_path / 'best_coco_bbox_mAP_50_epoch_200.pth',
        output_path / 'layout_model_checkpoint.enc'
    )


if __name__ == '__main__':
    main()
