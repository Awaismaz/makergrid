# middleware.py
class NewAccessTokenMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        new_access = request.META.get("HTTP_NEW_ACCESS")
        if new_access:
            response["new-access-token"] = new_access  # Frontend reads this
        return response
