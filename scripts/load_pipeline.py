from src.generation import QueryRouter
if __name__ == "__main__":
    router = QueryRouter()

    querie ="What are the side effects of the COVID-19 vaccine?"
    domain, ans = router.route(querie)
    print(domain)
    print(ans)