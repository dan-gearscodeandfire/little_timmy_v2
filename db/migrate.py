"""Apply database schema."""

import os
import asyncpg
import config


async def run():
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path) as f:
        sql = f.read()
    conn = await asyncpg.connect(config.DB_DSN)
    try:
        await conn.execute(sql)
        print("[DB] Schema applied successfully")
    finally:
        await conn.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
