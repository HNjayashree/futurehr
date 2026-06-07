# NexusHR 🚀
 
> **AI-Powered Human Resource Management System** — built for modern workplaces with machine learning at its core.
 
![Node.js](https://img.shields.io/badge/Node.js-24-green?logo=node.js)
![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue?logo=typescript)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
 
---
 
## What is NexusHR?
 
NexusHR is a full-stack, multi-role HR Management System that combines a production-grade TypeScript/React web app with a Python machine learning suite. It covers the complete employee lifecycle — from AI-powered recruitment and resume screening through to payroll, performance reviews, and predictive attrition analysis.
 
Built as a hackathon demo, it showcases how modern AI/ML can automate and augment every major HR workflow without human intervention.
 
---
 
## ✨ Features
 
### Full-Stack Web App
 
| Feature | Description |
|---|---|
| 🏠 **Multi-role Dashboard** | Admin sees company stats, headcount charts, activity feed; employees see personal attendance, leaves, payroll |
| 👥 **Employee Management** | Full CRUD — add, edit, deactivate employees with department assignment and status tracking |
| 🕐 **Attendance Tracking** | Clock-in/clock-out with daily status: present, late, absent, half-day, WFH |
| 🌴 **Leave Management** | Apply for leave, approve/reject requests, leave balance tracking |
| 💰 **Payroll** | Monthly payslip generation, deductions, tax calculation, download support |
| 📊 **Performance Reviews** | Review cycles with manager, peer, and self-assessment score breakdowns |
| 🤖 **AI Resume Screening** | OpenAI gpt-4o-mini automatically scores and ranks candidates |
| 🎙️ **AI Interview Sessions** | Streaming chat interface powered by OpenAI gpt-4o for preliminary candidate screening |
| 🔐 **Role-Based Access** | Admin, Senior Manager, HR Recruiter, Employee — each sees only what they need |
 
### Python ML Suite
 
| Module | Model(s) | Task | Performance |
|---|---|---|---|
| 📄 **Resume Screening** | Gradient Boosting + Random Forest + Logistic Regression | Binary classify: shortlist or reject | AUC ~0.98 |
| 📉 **Attrition Prediction** | RF + GB + LR Soft Voting Ensemble | Predict employees likely to leave | AUC ~0.99 |
| 🚨 **Attendance Anomaly** | Isolation Forest + DBSCAN | Detect abnormal attendance patterns | Consensus flagging |
| ⭐ **Performance Prediction** | GBR + SVR + Ridge + Logistic Regression | Predict scores + promotion likelihood | R² ~0.90 |
| 💳 **Payroll & Salary** | Isolation Forest + Random Forest Regressor | Flag payroll fraud + predict fair market salary | R² ~0.92 |
 
---
 
## 🗂️ Project Structure
 
```
nexushr/
│
├── artifacts/
│   ├── api-server/              # Express 5 REST API (TypeScript)
│   │   ├── src/routes/          # All route handlers
│   │   └── src/lib/openai.ts    # OpenAI client
│   └── hrms-web/                # React 19 + Vite frontend
│       ├── src/pages/           # All frontend pages
│       └── src/components/      # Shared components + layout
│
├── lib/
│   ├── db/src/schema/           # Drizzle ORM schema (PostgreSQL)
│   ├── api-spec/                # OpenAPI spec (source of truth)
│   ├── api-client-react/        # Generated React Query hooks
│   └── api-zod/                 # Generated Zod validation schemas
│
├── nexushr_ml/                  # Python ML Suite
│   ├── run_all.py               # Master runner — trains all 5 models
│   ├── requirements.txt
│   ├── data/
│   │   └── generate_data.py     # Synthetic HR data generator
│   └── models/
│       ├── resume_screening.py
│       ├── attrition_prediction.py
│       ├── attendance_anomaly.py
│       ├── performance_prediction.py
│       └── payroll_salary.py
│
├── package.json
├── pnpm-workspace.yaml
├── tsconfig.json
└── tsconfig.base.json
```
 
---
 
## 🚀 Getting Started
 
### Prerequisites
 
- Node.js 24+
- pnpm 9+
- PostgreSQL 16+
- Python 3.10+
- OpenAI API key
### 1. Clone the repo
 
```bash
git clone https://github.com/YOUR_USERNAME/nexushr.git
cd nexushr
```
 
### 2. Install dependencies
 
```bash
pnpm install
```
 
### 3. Set up environment variables
 
Create a `.env` file in the project root:
 
```env
DATABASE_URL=postgresql://user:password@localhost:5432/nexushr
OPENAI_API_KEY=sk-...
```
 
### 4. Set up the database
 
```bash
pnpm --filter @workspace/db run push
```
 
### 5. Run the development servers
 
In two separate terminals:
 
```bash
# Terminal 1 — API server (port 8080)
pnpm --filter @workspace/api-server run dev
 
# Terminal 2 — React frontend (port 22124)
pnpm --filter @workspace/hrms-web run dev
```
 
Open `http://localhost:22124` in your browser.
 
---
 
## 🐍 Running the ML Suite
 
```bash
cd nexushr_ml
pip install -r requirements.txt
python run_all.py
```
 
This trains all 5 ML models, prints evaluation metrics, and saves 6 analysis charts:
 
```
resume_screening_analysis.png   — ROC curves, feature importance, confusion matrix
attrition_analysis.png          — Precision-recall curves, risk factor rankings
attendance_anomalies.png        — PCA cluster plots, anomalous pattern scatter
performance_analysis.png        — Model comparison, actual vs predicted
payroll_anomalies.png           — Payroll fraud scatter plots
salary_analysis.png             — Salary equity gap distribution
```
 
### Inference example
 
```python
from nexushr_ml.models.resume_screening import train_resume_screener, screen_candidate
from nexushr_ml.data.generate_data import generate_resume_data
 
model, _ = train_resume_screener(generate_resume_data(500), plot=False)
 
result = screen_candidate(model, {
    "experience_years": 5,
    "education_level": 3,       # 1=Diploma, 2=Bachelor, 3=Master, 4=PhD
    "skills_count": 10,
    "relevant_skills": 8,
    "has_leadership_exp": 1,
    "certifications": 2,
    "gpa": 3.6,
    "portfolio_score": 7.8,
    "previous_companies": 2,
    "average_tenure_years": 2.5,
})
 
print(result)
# {'shortlist_probability': 0.82, 'recommendation': 'Shortlist', 'confidence': 'High'}
```
 
---
 
## 🔑 Demo Credentials
 
| Role | Email | Password |
|---|---|---|
| Admin | admin@nexushr.com | admin123 |
| Senior Manager | sarah.chen@nexushr.com | sarah123 |
| HR Recruiter | marcus.j@nexushr.com | marcus123 |
| Employee | emily.r@nexushr.com | emp123 |
 
> ⚠️ These are demo credentials. Passwords are stored in plaintext — replace with bcrypt before any production use.
 
---
 
## 🛠️ Tech Stack
 
### Web App
 
| Layer | Technology |
|---|---|
| Frontend | React 19, Vite 7, Wouter, Recharts, shadcn/ui, Tailwind CSS 4 |
| Backend | Express 5, Node.js 24, TypeScript 5.9 |
| Database | PostgreSQL + Drizzle ORM |
| Validation | Zod v4 |
| API Contract | OpenAPI spec → Orval codegen (React Query hooks + Zod schemas) |
| AI | OpenAI gpt-4o (interviews), gpt-4o-mini (resume screening) |
| Package Manager | pnpm workspaces |
| Build | esbuild (CJS bundle for API) |
 
### ML Suite
 
| Library | Version | Used For |
|---|---|---|
| scikit-learn | ≥1.4 | All ML models |
| pandas | ≥2.2 | Data manipulation |
| numpy | ≥1.26 | Numerical operations |
| matplotlib | ≥3.8 | Analysis charts |
 
---
 
## 🤖 ML Model Details
 
### Resume Screening
- **Algorithm**: Gradient Boosting Classifier (best: Logistic Regression pipeline)
- **Features**: experience, skills match ratio, education × GPA, career stability, portfolio score
- **Output**: shortlist probability (0–1), recommendation, confidence level
### Attrition Prediction
- **Algorithm**: Soft Voting Ensemble (Random Forest + Gradient Boosting + Logistic Regression)
- **Features**: satisfaction score, overtime hours, salary percentile by department, distance from office, tenure, promotions
- **Output**: attrition probability, risk tier (Low / Medium / High)
### Attendance Anomaly Detection
- **Algorithm**: Isolation Forest + DBSCAN (consensus — both must flag)
- **Features**: absent rate, late rate, WFH rate, avg clock-in hour, max consecutive absence streak
- **Output**: flagged employees with specific risk reasons
### Performance Prediction
- **Regression**: Gradient Boosting, Random Forest, SVR, Ridge → predicts composite review score
- **Classifier**: Logistic Regression → predicts promotion recommendation (binary)
- **Features**: training hours, projects delivered, goals completion %, manager/peer/self scores
### Payroll & Fair Salary
- **Anomaly**: Isolation Forest on payroll entries (overtime, bonuses, deductions, net pay)
- **Regression**: Random Forest Regressor → predicts market salary (R² ~0.92)
- **Equity analysis**: flags employees >15% below or above predicted market salary
---
 
## 📋 Available Scripts
 
```bash
# Install all packages
pnpm install
 
# Run full typecheck
pnpm run typecheck
 
# Build everything
pnpm run build
 
# Regenerate API client from OpenAPI spec
pnpm --filter @workspace/api-spec run codegen
 
# Push database schema changes
pnpm --filter @workspace/db run push
 
# Run ML suite
python nexushr_ml/run_all.py
```
 
---
 
## ⚠️ Important Notes
 
- **Auth tokens** use simple base64 encoding (`userId:email:timestamp`) — not production-safe. Replace with JWT for real deployments.
- **Passwords** are stored in plaintext for demo purposes only. Swap to bcrypt before going live.
- **OpenAI API key** is used directly — costs will apply based on usage.
- Always run `pnpm --filter @workspace/db run push` after any schema changes.
- Always run `pnpm --filter @workspace/api-spec run codegen` after editing the OpenAPI spec.
- Never run `pnpm run dev` at the workspace root — run per-artifact instead.
---
 
## 🤝 Contributing
 
1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request
---
 
## 📄 License
 
MIT — see [LICENSE](LICENSE) for details.
 
---
 
<div align="center">
  Built with ❤️ for the hackathon · NexusHR © 2025
</div>
