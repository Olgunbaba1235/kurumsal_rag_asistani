// Bellek içi PostgreSQL (PGlite) + pgvector'u Postgres wire protokolü üzerinden sunar.
// Kullanım: node server.mjs <port>   → hazır olunca stdout'a "READY <port>" yazar.
import { PGlite } from "@electric-sql/pglite";
import { vector } from "@electric-sql/pglite-pgvector";
import { PGLiteSocketServer } from "@electric-sql/pglite-socket";

const port = Number(process.argv[2] || 54329);
const db = await PGlite.create({ extensions: { vector } });
// PGlite tek bir backend sürecidir; tek bağlantı, oturum ayarlarının (SET LOCAL ROLE,
// app.tenant_id) bağlantılar arasında karışmamasını garanti eder.
const server = new PGLiteSocketServer({ db, port, host: "127.0.0.1", maxConnections: 1 });
await server.start();
console.log(`READY ${port}`);

const shutdown = async () => {
  try { await server.stop(); await db.close(); } finally { process.exit(0); }
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
