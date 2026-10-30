import fs from "fs";
import path from "path";
import { getDbPool } from "./db";

export async function runMigrations() {
  const pool = getDbPool();
  const schemaPath = path.join(process.cwd(), "src", "lib", "schema.sql");
  const sql = fs.readFileSync(schemaPath, "utf-8");
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const statements = sql
      .split(";")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    for (const stmt of statements) {
      await client.query(stmt);
    }
    await client.query("COMMIT");
    console.log("[MIGRATE] Database schema applied successfully.");
  } catch (err) {
    await client.query("ROLLBACK");
    console.error("[MIGRATE] Error applying database schema:", err);
    throw err;
  } finally {
    client.release();
  }
}

if (process.argv[1]?.endsWith("migrate.ts") || process.argv[1]?.endsWith("migrate.js")) {
  runMigrations()
    .then(() => process.exit(0))
    .catch((err) => {
      console.error(err);
      process.exit(1);
    });
}

