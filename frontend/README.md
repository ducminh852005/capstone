# CầuLôngStats frontend

React + TypeScript + Vite app for visualising badminton analysis results (player heatmaps,
shuttle trajectories, IN/OUT calls) produced by the backend in `../backend`.

Status: scaffold only. `src/App.tsx` is still the Vite template; nothing is wired to the API yet.

## Run

```cmd
npm install
npm run dev      # http://localhost:5173
npm run build
```

The backend (`uvicorn main:app --reload` in `../backend`) allows the origins listed in the
`CORS_ORIGINS` environment variable (default `http://localhost:5173`).

Project-wide development rules are in `../CLAUDE.md`.
