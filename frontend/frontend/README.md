# PolicyLens Frontend

Mobile-first Next.js prototype for the PolicyLens public-policy discussion platform.

## Status
This is a visual and interaction prototype using fictional local demo posts. It is not connected to authentication, a database, voting APIs, moderation services or the research pipeline.

## Stack
- Next.js App Router + React
- TypeScript strict mode
- Tailwind CSS v4
- Lucide React line icons
- DM Sans body type and Manrope display type
- CSS design tokens and light/dark themes

## Run locally
Requires Node.js 20.19+, 22.13+, or 24+.

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Open http://localhost:3000.

## Design direction
Editorial rather than generic admin UI: warm paper backgrounds, deep evergreen accents, restrained borders, stronger typographic hierarchy and documentary-style photography. Mobile uses horizontally scrollable filters and bottom navigation; desktop widens into a primary feed plus context rail.

## Routes
- `/` — feed, latest/trending/24-hour filters, search, categories, language selection and theme switch
- `/policy/[slug]` — detailed discussion, stance controls, comment composer and source-evidence placeholder
- `/discover` — topic directory
- `/about` — project principles and transparency
- `/login`, `/register` — UI-only forms

## Security and research boundaries
- Demo content and counts are fictional; never report them as research findings.
- Authentication forms are not connected. Do not use them in production until backend authentication exists.
- Client-only rate limits do not protect an API. Enforce limits in shared server infrastructure.
- Enforce one reaction per user/policy with a database uniqueness constraint.
- Do not rely on MAC addresses or invasive fingerprinting to prevent duplicate accounts.
- Label provisional community aggregates separately from validated research results.
- Do not replace research datasets, annotation worksheets, model artifacts or existing test files with demo content.
