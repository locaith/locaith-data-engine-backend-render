"""
Supabase Storage Service
Handles file storage using Supabase Storage (instead of local filesystem)
"""
import os
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# Get Supabase credentials
SUPABASE_URL = os.environ.get("URL_SUPABASE", "").strip().rstrip(',')
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "").strip()

# Storage bucket name
BUCKET_NAME = "documents"


class SupabaseStorageService:
    def __init__(self):
        self.client = None
        self.init_error = None
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            self.init_error = "Supabase credentials not set"
            print(f"[Supabase Storage] Warning: {self.init_error}")
            return
        
        try:
            from supabase import create_client
            self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
            print(f"[Supabase Storage] Initialized successfully")
            
            # Ensure bucket exists
            self._ensure_bucket()
        except Exception as e:
            self.init_error = str(e)
            print(f"[Supabase Storage] Failed to initialize: {e}")
    
    def _ensure_bucket(self):
        """Create storage bucket if it doesn't exist"""
        if not self.client:
            return
        
        try:
            # List existing buckets
            buckets = self.client.storage.list_buckets()
            bucket_names = [b.name for b in buckets]
            
            if BUCKET_NAME not in bucket_names:
                # Create the bucket (public for easy access)
                self.client.storage.create_bucket(
                    BUCKET_NAME,
                    options={"public": True}
                )
                print(f"[Supabase Storage] Created bucket: {BUCKET_NAME}")
            else:
                print(f"[Supabase Storage] Bucket exists: {BUCKET_NAME}")
        except Exception as e:
            print(f"[Supabase Storage] Bucket creation error (may already exist): {e}")
    
    def is_available(self) -> bool:
        """Check if storage service is available"""
        return self.client is not None
    
    async def upload_file(self, file_content: bytes, file_path: str, content_type: str = None) -> Tuple[bool, str]:
        """
        Upload file to Supabase Storage
        
        Args:
            file_content: File bytes
            file_path: Path in storage (e.g., "user123/doc.pdf")
            content_type: MIME type (optional)
        
        Returns:
            Tuple of (success, url_or_error)
        """
        if not self.is_available():
            return False, "Storage service not available"
        
        try:
            # Upload to Supabase Storage
            options = {}
            if content_type:
                options["content-type"] = content_type
            
            response = self.client.storage.from_(BUCKET_NAME).upload(
                file_path,
                file_content,
                file_options=options
            )
            
            # Get public URL
            public_url = self.client.storage.from_(BUCKET_NAME).get_public_url(file_path)
            
            print(f"[Supabase Storage] Uploaded: {file_path}")
            return True, public_url
            
        except Exception as e:
            error_msg = str(e)
            print(f"[Supabase Storage] Upload error: {error_msg}")
            
            # Handle duplicate file error
            if "Duplicate" in error_msg or "already exists" in error_msg.lower():
                # Try to get the existing file URL
                try:
                    public_url = self.client.storage.from_(BUCKET_NAME).get_public_url(file_path)
                    return True, public_url
                except:
                    pass
            
            return False, error_msg
    
    async def download_file(self, file_path: str) -> Tuple[bool, bytes]:
        """
        Download file from Supabase Storage
        
        Returns:
            Tuple of (success, content_or_error)
        """
        if not self.is_available():
            return False, b"Storage service not available"
        
        try:
            response = self.client.storage.from_(BUCKET_NAME).download(file_path)
            return True, response
        except Exception as e:
            print(f"[Supabase Storage] Download error: {e}")
            return False, str(e).encode()
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from Supabase Storage"""
        if not self.is_available():
            return False
        
        try:
            self.client.storage.from_(BUCKET_NAME).remove([file_path])
            print(f"[Supabase Storage] Deleted: {file_path}")
            return True
        except Exception as e:
            print(f"[Supabase Storage] Delete error: {e}")
            return False
    
    def get_public_url(self, file_path: str) -> Optional[str]:
        """Get public URL for a file"""
        if not self.is_available():
            return None
        
        try:
            return self.client.storage.from_(BUCKET_NAME).get_public_url(file_path)
        except Exception as e:
            print(f"[Supabase Storage] URL error: {e}")
            return None


# Singleton instance
supabase_storage = SupabaseStorageService()
