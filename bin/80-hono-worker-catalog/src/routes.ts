// API Routes for Worker Catalog

import { Hono } from "hono";
import { WORKERS } from "./data";

export const routes = new Hono<{ Bindings: Env }>();

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// INSTALL SCRIPT ENDPOINT
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * GET /install.sh
 * Serve rbee-keeper installation script
 * Usage: curl -fsSL https://install.rbee.dev | sh
 */
routes.get("/install.sh", async (c) => {
  try {
    const script = await c.env.ASSETS.fetch(
      new Request("http://placeholder/install.sh")
    );
    
    if (!script.ok) {
      return c.json({ error: "Install script not found" }, 404);
    }
    
    return new Response(script.body, {
      headers: { 
        "Content-Type": "text/plain",
        "Cache-Control": "public, max-age=300"  // 5 minutes
      }
    });
  } catch (error) {
    console.error("Failed to fetch install script:", error);
    return c.json({ error: "Failed to fetch install script" }, 500);
  }
});

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// WORKER CATALOG ENDPOINTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * GET /workers
 * List all available worker variants
 */
routes.get("/workers", (c) => {
  return c.json({ workers: WORKERS });
});

/**
 * GET /workers/:id
 * Get a specific worker by ID
 */
routes.get("/workers/:id", (c) => {
  const id = c.req.param("id");
  const worker = WORKERS.find(w => w.id === id);
  
  if (!worker) {
    return c.json({ error: "Worker not found" }, 404);
  }
  
  return c.json(worker);
});

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PKGBUILD ENDPOINTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * GET /workers/:id/PKGBUILD
 * Serve PKGBUILD file for a specific worker
 */
routes.get("/workers/:id/PKGBUILD", async (c) => {
  const id = c.req.param("id");
  
  // Verify worker exists
  const worker = WORKERS.find(w => w.id === id);
  if (!worker) {
    return c.json({ error: "Worker not found" }, 404);
  }
  
  try {
    // TEAM-453: Check if ASSETS binding exists (not available in test environment)
    if (!c.env?.ASSETS) {
      return c.json({ error: "ASSETS binding not available in test environment" }, 500);
    }
    
    // Fetch PKGBUILD from assets
    const pkgbuild = await c.env.ASSETS.fetch(
      new Request(`http://placeholder/pkgbuilds/${id}.PKGBUILD`)
    );
    
    if (!pkgbuild.ok) {
      return c.json({ error: "PKGBUILD not found" }, 404);
    }
    
    return new Response(pkgbuild.body, {
      headers: { 
        "Content-Type": "text/plain",
        "Cache-Control": "public, max-age=3600"
      }
    });
  } catch (error) {
    console.error(`Failed to fetch PKGBUILD for ${id}:`, error);
    return c.json({ error: "Failed to fetch PKGBUILD" }, 500);
  }
});
