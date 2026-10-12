import os
import logging
import re
import psycopg
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("carworld.database")

def search_car_database(query: str):
    try:
        # Match each meaningful search term across make, model, and variant.
        # This supports both partial names ("Scorpio") and combined names
        # ("Mahindra Scorpio") even though those values live in separate fields.
        ignored_terms = {
            "a", "an", "and", "are", "about", "car", "cars", "cost",
            "current", "currently", "database", "details", "does", "find",
            "for", "give", "in", "is", "lakh", "latest", "list", "me",
            "mileage", "model", "models", "of", "please", "price", "prices",
            "search", "show", "spec", "specs", "tell", "the", "under",
            "variant", "variants", "what",
        }
        terms = [
            term for term in re.findall(r"[\w]+", str(query), flags=re.UNICODE)
            if term.casefold() not in ignored_terms and not term.isdigit()
        ]
        if not terms:
            return {"status": "ok", "count": 0, "results": []}

        term_predicates = [
            "(make ILIKE %s OR model ILIKE %s OR variant ILIKE %s)"
            for _term in terms
        ]
        sql = f"""
            SELECT make, model, variant, ex_showroom_price,
                   fuel_type, transmission, body_type
            FROM cars
            WHERE {' AND '.join(term_predicates)}
            LIMIT 10;
        """
        params = tuple(pattern for term in terms for pattern in (f"%{term}%",) * 3)

        with psycopg.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            dbname=os.getenv("POSTGRES_DB"),
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return {
            "status": "ok",
            "count": len(rows),
            "results": [
                {
                    "make": row[0],
                    "model": row[1],
                    "variant": row[2],
                    "ex_showroom_price": row[3],
                    "fuel_type": row[4],
                    "transmission": row[5],
                    "body_type": row[6],
                }
                for row in rows
            ],
        }

    except Exception as error:
        # Database driver errors can contain connection details; keep logs and
        # tool output free of DSNs, hostnames, and credential-adjacent values.
        logger.error("Database search failed (%s).", type(error).__name__)
        return {"status": "error", "message": "Database search is temporarily unavailable."}
