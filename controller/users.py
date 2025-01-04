from fastapi import APIRouter

router = APIRouter(
    prefix="/users",
    tags=["items"],
    responses={404: {"description": "Not Found"}},
)

@router.get("/{user_id}")
def read_item(user_id: int):
    return {"user_id" : user_id}
