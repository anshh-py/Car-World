
import os
import sys

import psycopg
from psycopg import sql
from dotenv import load_dotenv


# Load environment variables from .env in this project folder
load_dotenv()


def inspect_database():
    conninfo = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "dbname": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "connect_timeout": 5,
    }

    required = ("dbname", "user", "password")
    missing = [key for key in required if not conninfo.get(key)]

    if missing:
        print(
            "Missing PostgreSQL configuration: "
            + ", ".join(missing)
        )
        print("Check your .env file.")
        sys.exit(1)

    try:
        print("Connecting to PostgreSQL...")

        # Read-only session: this script only inspects the database.
        with psycopg.connect(
            **conninfo,
            options="-c default_transaction_read_only=on",
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_database(), current_user")
                database, username = cur.fetchone()

                print(f"\nConnected successfully!")
                print(f"Database: {database}")
                print(f"User: {username}")

                cur.execute("""
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE table_type = 'BASE TABLE'
                      AND table_schema NOT IN (
                          'pg_catalog',
                          'information_schema'
                      )
                    ORDER BY table_schema, table_name;
                """)

                tables = cur.fetchall()

                if not tables:
                    print("\nNo tables found in accessible schemas.")
                    return

                print(f"\nTables found: {len(tables)}")

                total_rows = 0

                for schema, table in tables:
                    print(f"\n{'=' * 55}")
                    print(f"TABLE: {schema}.{table}")

                    cur.execute("""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns
                        WHERE table_schema = %s
                          AND table_name = %s
                        ORDER BY ordinal_position;
                    """, (schema, table))

                    columns = cur.fetchall()

                    for column, data_type, nullable in columns:
                        print(
                            f"  {column}: {data_type} "
                            f"(nullable: {nullable})"
                        )

                    # Safely quote schema and table identifiers.
                    count_query = sql.SQL(
                        "SELECT COUNT(*) FROM {}.{}"
                    ).format(
                        sql.Identifier(schema),
                        sql.Identifier(table),
                    )

                    cur.execute(count_query)
                    row_count = cur.fetchone()[0]
                    total_rows += row_count

                    print(f"  ROW COUNT: {row_count}")

                print(f"\n{'=' * 55}")
                print(f"Total tables: {len(tables)}")
                print(f"Total rows across tables: {total_rows}")
                print("\nDatabase inspection completed.")

    except psycopg.OperationalError as exc:
        print("\nCould not connect to PostgreSQL.")
        print("Check that PostgreSQL is running and your .env")
        print("contains the correct host, port, database and credentials.")
        print(f"Details: {exc}")
        sys.exit(1)

    except psycopg.Error as exc:
        print(f"\nPostgreSQL error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    inspect_database()
