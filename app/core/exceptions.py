"""
Custom exception classes for AI modification service.

These exceptions map to proper HTTP status codes that should be returned to clients.
"""

from fastapi import HTTPException, status


class AIAuthenticationError(HTTPException):
    """Raised when AI service authentication fails (401 Unauthorized)"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=detail
        )


class AIValidationError(HTTPException):
    """Raised when request validation fails (400 Bad Request)"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST, detail=detail
        )


class AIServiceError(HTTPException):
    """Raised when AI service encounters internal errors (500 Internal Server Error)"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail
        )


class AIRateLimitError(HTTPException):
    """Raised when rate limit is exceeded (429 Too Many Requests)"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail
        )
