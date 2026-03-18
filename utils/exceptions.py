"""DEEPVISION Exceptions Module"""


class DeepVisionError(Exception):
    """Base exception for DEEPVISION"""
    pass


class InvalidFileError(DeepVisionError):
    """Exception raised for invalid file formats or corrupted files"""
    pass


class ProcessingError(DeepVisionError):
    """Exception raised during processing failures"""
    pass


class ModelLoadError(DeepVisionError):
    """Exception raised when model loading fails"""
    pass


class UnsupportedFormatError(DeepVisionError):
    """Exception raised for unsupported file formats"""
    pass


class ConfigurationError(DeepVisionError):
    """Exception raised for configuration issues"""
    pass


class DataError(DeepVisionError):
    """Exception raised for data-related issues"""
    pass


class TrainingError(DeepVisionError):
    """Exception raised during training failures"""
    pass


class ValidationError(DeepVisionError):
    """Exception raised for validation failures"""
    pass