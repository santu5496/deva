"""
Utility functions for the Leprosy Detection application
"""

import json
import os
import platform
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase

def normalize_path(path):
    """
    Normalize file path for cross-platform compatibility.
    Ensures paths use forward slashes regardless of operating system.
    
    Args:
        path (str): File path to normalize
        
    Returns:
        str: Normalized path with forward slashes
    """
    # Replace backslashes with forward slashes (for Windows compatibility)
    normalized = path.replace('\\', '/')
    
    # Remove any duplicate slashes
    while '//' in normalized:
        normalized = normalized.replace('//', '/')
        
    return normalized

def join_paths(*paths):
    """
    Join path components using forward slashes for cross-platform compatibility.
    
    Args:
        *paths: Path components to join
        
    Returns:
        str: Joined path with forward slashes
    """
    # Join paths using os.path.join which respects the current OS
    joined = os.path.join(*paths)
    
    # Normalize to forward slashes for consistency
    return normalize_path(joined)

def serialize_model(obj):
    """
    Serialize SQLAlchemy model object to dictionary
    
    Args:
        obj: SQLAlchemy model instance
        
    Returns:
        dict: Dictionary representation of the model
    """
    if isinstance(obj, DeclarativeBase):
        data = {}
        for column in obj.__table__.columns:
            value = getattr(obj, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            data[column.name] = value
        return data
    
    if isinstance(obj, datetime):
        return obj.isoformat()
        
    return obj

def to_json(data):
    """
    Convert data to JSON with custom serializer for SQLAlchemy models
    
    Args:
        data: Data to serialize
        
    Returns:
        str: JSON string
    """
    return json.dumps(data, default=serialize_model)

def model_to_dict(obj):
    """
    Convert SQLAlchemy model to dictionary
    
    Args:
        obj: SQLAlchemy model instance
        
    Returns:
        dict: Dictionary representation of the model
    """
    return serialize_model(obj)