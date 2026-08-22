import { loadEnvConfig } from "@next/env";

const projectDir = process.cwd();
loadEnvConfig(projectDir);

export const API_URL = process.env.API_URL || "http://localhost:8000";
