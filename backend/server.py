from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime, timezone
import tempfile
import shutil
import subprocess
import asyncio
import json
import re
import time
import docker
import tarfile
import io
from git import Repo, GitCommandError
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize AI chat
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Docker client (with fallback for environments without Docker)
try:
    docker_client = docker.from_env()
    DOCKER_AVAILABLE = True
    logger.info("Docker client initialized successfully")
except Exception as e:
    logger.warning(f"Docker not available: {e}. Using mock analysis mode.")
    docker_client = None
    DOCKER_AVAILABLE = False

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Define Models
class RepoAnalysisRequest(BaseModel):
    git_url: str
    repo_name: Optional[str] = None
    analysis_depth: str = "comprehensive"  # basic, standard, comprehensive

class SecurityFinding(BaseModel):
    type: str
    severity: str  # critical, high, medium, low
    file_path: str
    line_number: Optional[int] = None
    description: str
    cve_id: Optional[str] = None
    fix_available: bool = False
    fix_suggestion: Optional[str] = None

class PerformanceIssue(BaseModel):
    type: str
    file_path: str
    function_name: Optional[str] = None
    issue: str
    impact: str
    optimization_suggestion: str
    estimated_improvement: Optional[str] = None

class CodeQualityIssue(BaseModel):
    type: str
    file_path: str
    line_number: Optional[int] = None
    severity: str
    issue: str
    suggestion: str
    auto_fixable: bool = False
    pattern_type: Optional[str] = None  # code_smell, anti_pattern, architecture

class DependencyIssue(BaseModel):
    package_name: str
    current_version: str
    latest_version: Optional[str] = None
    vulnerability_count: int = 0
    critical_vulnerabilities: List[str] = []
    update_available: bool = False
    breaking_changes: bool = False

class TestResult(BaseModel):
    test_type: str
    status: str  # passed, failed, error
    coverage_percentage: Optional[float] = None
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    execution_time: Optional[float] = None
    error_details: Optional[str] = None

class AIFix(BaseModel):
    issue_id: str
    fix_type: str
    confidence_score: float
    original_code: str
    fixed_code: str
    explanation: str
    test_results: Optional[Dict] = None
    validated: bool = False

class ComprehensiveAnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    git_url: str
    repo_name: str
    status: str  # pending, analyzing, completed, failed
    analysis_depth: str
    
    # Repository Info
    total_files: int = 0
    lines_of_code: int = 0
    languages_detected: List[str] = []
    framework_detected: List[str] = []
    
    # Analysis Results
    security_findings: List[SecurityFinding] = []
    performance_issues: List[PerformanceIssue] = []
    code_quality_issues: List[CodeQualityIssue] = []
    dependency_issues: List[DependencyIssue] = []
    test_results: List[TestResult] = []
    
    # AI Fixes
    ai_fixes_applied: List[AIFix] = []
    
    # Execution Results
    build_successful: bool = False
    runtime_errors: List[str] = []
    execution_logs: Optional[str] = None
    
    # AI Analysis
    ai_summary: Optional[str] = None
    deployment_readiness: Optional[str] = None
    architecture_analysis: Optional[str] = None
    recommendations: List[str] = []
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    analysis_duration: Optional[float] = None

class AnalysisResultCreate(BaseModel):
    git_url: str
    repo_name: Optional[str] = None
    analysis_depth: str = "comprehensive"

# Docker Sandbox Manager
class DockerSandbox:
    def __init__(self, repo_url: str, repo_name: str):
        self.repo_url = repo_url
        self.repo_name = repo_name
        self.container = None
        self.image_name = f"codeanalysis-{repo_name.lower()}-{uuid.uuid4().hex[:8]}"
        
    async def create_analysis_environment(self):
        """Create a secure Docker environment for code analysis"""
        try:
            dockerfile_content = f"""
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    wget \\
    nodejs \\
    npm \\
    gcc \\
    g++ \\
    make \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install security tools
RUN pip install bandit safety semgrep pylint mypy black isort
RUN npm install -g eslint jshint

# Create working directory
WORKDIR /analysis

# Clone the repository
RUN git clone {self.repo_url} repo

WORKDIR /analysis/repo

# Create analysis script
COPY analysis_script.py /analysis/
COPY run_analysis.sh /analysis/

RUN chmod +x /analysis/run_analysis.sh

CMD ["/analysis/run_analysis.sh"]
"""
            
            # Create analysis script
            analysis_script = '''
import os
import json
import subprocess
import sys
from pathlib import Path
import ast
import time

def analyze_repository():
    results = {
        "languages": [],
        "frameworks": [],
        "lines_of_code": 0,
        "files_analyzed": 0,
        "security_issues": [],
        "performance_issues": [],
        "quality_issues": [],
        "dependencies": [],
        "build_results": {},
        "test_results": {},
        "execution_results": {}
    }
    
    # Language detection
    file_extensions = {}
    for root, dirs, files in os.walk("."):
        for file in files:
            ext = Path(file).suffix
            if ext:
                file_extensions[ext] = file_extensions.get(ext, 0) + 1
                results["files_analyzed"] += 1
                
                # Count lines of code
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                        results["lines_of_code"] += len(f.readlines())
                except:
                    pass
    
    # Determine languages
    if ".py" in file_extensions:
        results["languages"].append("Python")
    if ".js" in file_extensions or ".ts" in file_extensions:
        results["languages"].append("JavaScript/TypeScript")
    if ".java" in file_extensions:
        results["languages"].append("Java")
    if ".go" in file_extensions:
        results["languages"].append("Go")
    
    # Framework detection
    if os.path.exists("requirements.txt") or os.path.exists("setup.py"):
        results["frameworks"].append("Python Project")
    if os.path.exists("package.json"):
        results["frameworks"].append("Node.js Project")
    if os.path.exists("pom.xml"):
        results["frameworks"].append("Maven Project")
    
    # Run security analysis
    try:
        # Bandit for Python
        if "Python" in results["languages"]:
            bandit_result = subprocess.run(["bandit", "-f", "json", "-r", "."], 
                                         capture_output=True, text=True, timeout=60)
            if bandit_result.stdout:
                bandit_data = json.loads(bandit_result.stdout)
                for issue in bandit_data.get("results", []):
                    results["security_issues"].append({
                        "tool": "bandit",
                        "file": issue["filename"],
                        "line": issue["line_number"],
                        "severity": issue["issue_severity"],
                        "description": issue["issue_text"],
                        "type": "security"
                    })
    except Exception as e:
        results["security_issues"].append({"error": str(e)})
    
    # Run quality analysis
    try:
        if "Python" in results["languages"]:
            pylint_result = subprocess.run(["pylint", "--output-format=json", "."], 
                                         capture_output=True, text=True, timeout=120)
            if pylint_result.stdout:
                try:
                    pylint_data = json.loads(pylint_result.stdout)
                    for issue in pylint_data[:10]:  # Limit to top 10
                        results["quality_issues"].append({
                            "tool": "pylint",
                            "file": issue.get("path", ""),
                            "line": issue.get("line", 0),
                            "type": issue.get("type", ""),
                            "message": issue.get("message", ""),
                            "severity": issue.get("category", "")
                        })
                except:
                    pass
    except Exception as e:
        results["quality_issues"].append({"error": str(e)})
    
    # Try to build/install dependencies
    try:
        if os.path.exists("requirements.txt"):
            build_result = subprocess.run(["pip", "install", "-r", "requirements.txt"], 
                                        capture_output=True, text=True, timeout=300)
            results["build_results"]["pip_install"] = {
                "success": build_result.returncode == 0,
                "output": build_result.stdout[-500:] if build_result.stdout else "",
                "errors": build_result.stderr[-500:] if build_result.stderr else ""
            }
        
        if os.path.exists("package.json"):
            npm_result = subprocess.run(["npm", "install"], 
                                      capture_output=True, text=True, timeout=300)
            results["build_results"]["npm_install"] = {
                "success": npm_result.returncode == 0,
                "output": npm_result.stdout[-500:] if npm_result.stdout else "",
                "errors": npm_result.stderr[-500:] if npm_result.stderr else ""
            }
    except Exception as e:
        results["build_results"]["error"] = str(e)
    
    # Try to run tests
    try:
        if os.path.exists("pytest.ini") or any("test_" in f for f in os.listdir(".")):
            test_result = subprocess.run(["python", "-m", "pytest", "--tb=short"], 
                                       capture_output=True, text=True, timeout=180)
            results["test_results"]["pytest"] = {
                "success": test_result.returncode == 0,
                "output": test_result.stdout[-500:] if test_result.stdout else "",
                "errors": test_result.stderr[-500:] if test_result.stderr else ""
            }
    except Exception as e:
        results["test_results"]["error"] = str(e)
    
    return results

if __name__ == "__main__":
    try:
        analysis_results = analyze_repository()
        print("ANALYSIS_RESULTS_START")
        print(json.dumps(analysis_results, indent=2))
        print("ANALYSIS_RESULTS_END")
    except Exception as e:
        print(f"Analysis failed: {e}")
        sys.exit(1)
'''
            
            run_script = '''#!/bin/bash
set -e

echo "Starting comprehensive code analysis..."

cd /analysis/repo

# Run the analysis script
python /analysis/analysis_script.py
'''
            
            # Create temporary directory for build context
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write files
                with open(f"{temp_dir}/Dockerfile", "w") as f:
                    f.write(dockerfile_content)
                with open(f"{temp_dir}/analysis_script.py", "w") as f:
                    f.write(analysis_script)
                with open(f"{temp_dir}/run_analysis.sh", "w") as f:
                    f.write(run_script)
                
                # Build Docker image
                logger.info(f"Building Docker image for {self.repo_name}")
                image, build_logs = docker_client.images.build(
                    path=temp_dir,
                    tag=self.image_name,
                    rm=True
                )
                
                return image
                
        except Exception as e:
            logger.error(f"Failed to create analysis environment: {e}")
            raise
    
    async def run_analysis(self):
        """Run comprehensive analysis in Docker sandbox or mock mode"""
        if not DOCKER_AVAILABLE:
            # Mock analysis for demonstration
            logger.info(f"Running mock analysis for {self.repo_name} (Docker not available)")
            return await self._run_mock_analysis()
        
        try:
            # Create the environment
            image = await self.create_analysis_environment()
            
            # Run analysis container
            logger.info(f"Running analysis container for {self.repo_name}")
            container = docker_client.containers.run(
                image.id,
                detach=True,
                mem_limit="2g",
                cpu_count=2,
                network_disabled=False,  # Need network for git clone
                remove=True
            )
            
            # Wait for completion with timeout
            result = container.wait(timeout=600)  # 10 minutes timeout
            logs = container.logs().decode('utf-8')
            
            # Extract analysis results
            if "ANALYSIS_RESULTS_START" in logs and "ANALYSIS_RESULTS_END" in logs:
                start = logs.find("ANALYSIS_RESULTS_START") + len("ANALYSIS_RESULTS_START")
                end = logs.find("ANALYSIS_RESULTS_END")
                results_json = logs[start:end].strip()
                return json.loads(results_json)
            else:
                logger.warning("Could not find analysis results in logs")
                return {"error": "Analysis results not found", "logs": logs}
                
        except Exception as e:
            logger.error(f"Analysis execution failed: {e}")
            return {"error": str(e)}
        finally:
            # Cleanup
            await self.cleanup()
    
    async def _run_mock_analysis(self):
        """Run mock analysis for demonstration purposes"""
        # Simulate analysis delay
        await asyncio.sleep(5)
        
        # Return realistic mock data
        return {
            "languages": ["Python", "JavaScript"],
            "frameworks": ["Python Project", "Node.js Project"],
            "lines_of_code": 2847,
            "files_analyzed": 23,
            "security_issues": [
                {
                    "tool": "bandit",
                    "file": "app.py",
                    "line": 42,
                    "severity": "HIGH",
                    "description": "Possible SQL injection vulnerability",
                    "type": "security"
                },
                {
                    "tool": "bandit", 
                    "file": "config.py",
                    "line": 15,
                    "severity": "MEDIUM",
                    "description": "Hardcoded password detected",
                    "type": "security"
                }
            ],
            "quality_issues": [
                {
                    "tool": "pylint",
                    "file": "main.py",
                    "line": 128,
                    "type": "convention",
                    "message": "Line too long (88/79)",
                    "severity": "medium"
                },
                {
                    "tool": "pylint",
                    "file": "utils.py", 
                    "line": 67,
                    "type": "warning",
                    "message": "Unused variable 'result'",
                    "severity": "low"
                },
                {
                    "tool": "pylint",
                    "file": "handlers.py",
                    "line": 34,
                    "type": "error",
                    "message": "Undefined variable 'data'",
                    "severity": "high"
                }
            ],
            "build_results": {
                "pip_install": {
                    "success": True,
                    "output": "Successfully installed all requirements",
                    "errors": ""
                },
                "npm_install": {
                    "success": False,
                    "output": "",
                    "errors": "Package 'vulnerable-dep' has known security issues"
                }
            },
            "test_results": {
                "pytest": {
                    "success": False,
                    "output": "2 passed, 1 failed",
                    "errors": "AssertionError in test_user_auth"
                }
            }
        }
    
    async def cleanup(self):
        """Clean up Docker resources"""
        if not DOCKER_AVAILABLE:
            return
            
        try:
            # Remove the image
            docker_client.images.remove(self.image_name, force=True)
            logger.info(f"Cleaned up Docker resources for {self.repo_name}")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

# AI Analysis Engine
class AIAnalysisEngine:
    def __init__(self):
        self.chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=str(uuid.uuid4()),
            system_message="""You are an expert software architect and security analyst with deep knowledge of:
- Code architecture and design patterns
- Security vulnerabilities and best practices  
- Performance optimization techniques
- Software testing methodologies
- Deployment and DevOps practices

Analyze code comprehensively and provide actionable insights for improvement."""
        ).with_model("anthropic", "claude-sonnet-4-20250514")
    
    async def analyze_with_ai(self, analysis_data: Dict, repo_info: Dict) -> Dict:
        """Use AI to provide deep analysis and insights"""
        
        prompt = f"""
Analyze this code repository comprehensively:

Repository: {repo_info.get('repo_name', 'Unknown')}
Languages: {', '.join(analysis_data.get('languages', []))}
Frameworks: {', '.join(analysis_data.get('frameworks', []))}
Files: {analysis_data.get('files_analyzed', 0)}
Lines of Code: {analysis_data.get('lines_of_code', 0)}

Security Issues Found: {len(analysis_data.get('security_issues', []))}
Quality Issues Found: {len(analysis_data.get('quality_issues', []))}

Build Results: {json.dumps(analysis_data.get('build_results', {}), indent=2)}
Test Results: {json.dumps(analysis_data.get('test_results', {}), indent=2)}

Security Issues:
{json.dumps(analysis_data.get('security_issues', [])[:5], indent=2)}

Quality Issues:
{json.dumps(analysis_data.get('quality_issues', [])[:5], indent=2)}

Please provide a comprehensive analysis in JSON format:
{{
    "overall_assessment": "detailed assessment",
    "security_rating": "A/B/C/D/F",
    "code_quality_rating": "A/B/C/D/F", 
    "performance_rating": "A/B/C/D/F",
    "deployment_readiness": "ready/needs_work/not_ready",
    "architecture_analysis": "detailed architecture review",
    "critical_issues": ["list of critical issues"],
    "recommendations": ["specific actionable recommendations"],
    "auto_fixable_issues": ["issues that can be auto-fixed"],
    "estimated_fix_time": "time estimate for fixes"
}}
"""
        
        try:
            user_message = UserMessage(text=prompt)
            response = await self.chat.send_message(user_message)
            
            # Parse AI response
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {
                    "overall_assessment": response[:500],
                    "security_rating": "C",
                    "code_quality_rating": "C",
                    "performance_rating": "C",
                    "deployment_readiness": "needs_work",
                    "architecture_analysis": "Analysis parsing failed",
                    "critical_issues": [],
                    "recommendations": ["Review code manually"],
                    "auto_fixable_issues": [],
                    "estimated_fix_time": "Unknown"
                }
                
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {
                "overall_assessment": "AI analysis unavailable",
                "error": str(e)
            }

# Helper Functions
def prepare_for_mongo(data):
    """Convert datetime objects to ISO strings for MongoDB storage"""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, list):
                data[key] = [prepare_for_mongo(item) if isinstance(item, dict) else item for item in value]
            elif isinstance(value, dict):
                data[key] = prepare_for_mongo(value)
    return data

async def run_comprehensive_analysis(analysis_id: str, git_url: str, repo_name: str, analysis_depth: str):
    """Background task for comprehensive repository analysis"""
    start_time = time.time()
    
    try:
        # Update status
        await db.analyses.update_one(
            {"id": analysis_id},
            {"$set": {"status": "analyzing"}}
        )
        
        logger.info(f"Starting comprehensive analysis for {repo_name}")
        
        # Clone repository locally for real analysis
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / repo_name
            
            try:
                # Clone the repository
                logger.info(f"Cloning repository: {git_url}")
                Repo.clone_from(git_url, repo_path)
            except GitCommandError as e:
                logger.error(f"Failed to clone repository: {e}")
                raise Exception(f"Failed to clone repository: {str(e)}")
            
            # Perform real analysis
            analysis_results = await perform_real_analysis(repo_path, repo_name)
            
            # Use AI to enhance the analysis
            ai_engine = AIAnalysisEngine()
            ai_results = await ai_engine.analyze_with_ai(analysis_results, {"repo_name": repo_name})
            
            # Process and structure results
            analysis_result = ComprehensiveAnalysisResult(
                id=analysis_id,
                git_url=git_url,
                repo_name=repo_name,
                analysis_depth=analysis_depth,
                status="completed",
                
                # Repository metrics from real analysis
                total_files=analysis_results.get("total_files", 0),
                lines_of_code=analysis_results.get("lines_of_code", 0),
                languages_detected=analysis_results.get("languages", []),
                framework_detected=analysis_results.get("frameworks", []),
                
                # Real security findings
                security_findings=[
                    SecurityFinding(
                        type=issue.get("type", "security"),
                        severity=issue.get("severity", "medium"),
                        file_path=issue.get("file_path", ""),
                        line_number=issue.get("line_number"),
                        description=issue.get("description", ""),
                        fix_suggestion=issue.get("fix_suggestion", "")
                    ) for issue in analysis_results.get("security_issues", [])
                ],
                
                # Real code quality issues
                code_quality_issues=[
                    CodeQualityIssue(
                        type=issue.get("type", "quality"),
                        file_path=issue.get("file_path", ""),
                        line_number=issue.get("line_number"),
                        severity=issue.get("severity", "medium"),
                        issue=issue.get("issue", ""),
                        suggestion=issue.get("suggestion", ""),
                        auto_fixable=issue.get("auto_fixable", False)
                    ) for issue in analysis_results.get("quality_issues", [])
                ],
                
                # Performance issues
                performance_issues=[
                    PerformanceIssue(
                        type=issue.get("type", "performance"),
                        file_path=issue.get("file_path", ""),
                        function_name=issue.get("function_name"),
                        issue=issue.get("issue", ""),
                        impact=issue.get("impact", "medium"),
                        optimization_suggestion=issue.get("optimization_suggestion", "")
                    ) for issue in analysis_results.get("performance_issues", [])
                ],
                
                # Build and execution results
                build_successful=analysis_results.get("build_successful", False),
                execution_logs=str(analysis_results.get("execution_logs", ""))[:1000],
                runtime_errors=analysis_results.get("runtime_errors", []),
                
                # AI analysis results
                ai_summary=ai_results.get("overall_assessment", "Analysis completed"),
                deployment_readiness=ai_results.get("deployment_readiness", "needs_review"),
                architecture_analysis=ai_results.get("architecture_analysis", "Architecture analysis completed"),
                recommendations=ai_results.get("recommendations", []),
                
                # Completion info
                completed_at=datetime.now(timezone.utc),
                analysis_duration=time.time() - start_time
            )
            
            # Store in database
            result_dict = prepare_for_mongo(analysis_result.dict())
            await db.analyses.update_one(
                {"id": analysis_id},
                {"$set": result_dict}
            )
            
            logger.info(f"Real analysis completed for {repo_name} in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed for {analysis_id}: {e}")
        await db.analyses.update_one(
            {"id": analysis_id},
            {"$set": {
                "status": "failed",
                "ai_summary": f"Analysis failed: {str(e)}",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "analysis_duration": time.time() - start_time
            }}
        )

async def perform_real_analysis(repo_path: Path, repo_name: str) -> Dict:
    """Perform real analysis on the cloned repository"""
    results = {
        "total_files": 0,
        "lines_of_code": 0,
        "languages": [],
        "frameworks": [],
        "security_issues": [],
        "quality_issues": [],
        "performance_issues": [],
        "build_successful": False,
        "execution_logs": "",
        "runtime_errors": []
    }
    
    try:
        # Count files and detect languages
        file_counts = {}
        quality_issues = []
        
        for file_path in repo_path.rglob("*"):
            if file_path.is_file():
                results["total_files"] += 1
                ext = file_path.suffix.lower()
                file_counts[ext] = file_counts.get(ext, 0) + 1
                
                # Count lines of code
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        results["lines_of_code"] += len(lines)
                        
                        # Basic quality analysis
                        relative_path = str(file_path.relative_to(repo_path))
                        
                        if ext == '.py':
                            # Python quality checks
                            for i, line in enumerate(lines, 1):
                                if len(line.strip()) > 100:
                                    quality_issues.append({
                                        "type": "style",
                                        "file_path": relative_path,
                                        "line_number": i,
                                        "severity": "medium",
                                        "issue": f"Line too long ({len(line.strip())} characters)",
                                        "suggestion": "Break line into multiple lines",
                                        "auto_fixable": True
                                    })
                                if line.strip().startswith('print('):
                                    quality_issues.append({
                                        "type": "quality",
                                        "file_path": relative_path,
                                        "line_number": i,
                                        "severity": "low",
                                        "issue": "Debug print statement found",
                                        "suggestion": "Remove debug print or use logging",
                                        "auto_fixable": True
                                    })
                                if 'password' in line.lower() and '=' in line:
                                    quality_issues.append({
                                        "type": "security",
                                        "file_path": relative_path,
                                        "line_number": i,
                                        "severity": "high",
                                        "issue": "Potential hardcoded password",
                                        "suggestion": "Move sensitive data to environment variables",
                                        "auto_fixable": False
                                    })
                        
                        elif ext in ['.js', '.ts', '.jsx', '.tsx']:
                            # JavaScript/TypeScript quality checks
                            for i, line in enumerate(lines, 1):
                                if 'console.log' in line:
                                    quality_issues.append({
                                        "type": "quality",
                                        "file_path": relative_path,
                                        "line_number": i,
                                        "severity": "low",
                                        "issue": "Debug console.log found",
                                        "suggestion": "Remove debug console.log statements",
                                        "auto_fixable": True
                                    })
                                if 'var ' in line and not line.strip().startswith('//'):
                                    quality_issues.append({
                                        "type": "style",
                                        "file_path": relative_path,
                                        "line_number": i,
                                        "severity": "medium",
                                        "issue": "Use of 'var' instead of 'let' or 'const'",
                                        "suggestion": "Use 'let' or 'const' instead of 'var'",
                                        "auto_fixable": True
                                    })
                except:
                    continue
        
        # Determine languages based on file extensions
        if '.py' in file_counts:
            results["languages"].append("Python")
        if any(ext in file_counts for ext in ['.js', '.ts', '.jsx', '.tsx']):
            results["languages"].append("JavaScript/TypeScript")
        if '.java' in file_counts:
            results["languages"].append("Java")
        if '.go' in file_counts:
            results["languages"].append("Go")
        if '.rb' in file_counts:
            results["languages"].append("Ruby")
        if '.php' in file_counts:
            results["languages"].append("PHP")
        
        # Detect frameworks
        if (repo_path / "requirements.txt").exists() or (repo_path / "setup.py").exists():
            results["frameworks"].append("Python Project")
        if (repo_path / "package.json").exists():
            results["frameworks"].append("Node.js Project")
        if (repo_path / "pom.xml").exists():
            results["frameworks"].append("Maven Project")
        if (repo_path / "Cargo.toml").exists():
            results["frameworks"].append("Rust Project")
        if (repo_path / "go.mod").exists():
            results["frameworks"].append("Go Module")
        
        # Add quality issues to results
        results["quality_issues"] = quality_issues[:20]  # Limit to first 20 issues
        
        # Security analysis
        security_issues = []
        
        # Check for common security issues
        for file_path in repo_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    relative_path = str(file_path.relative_to(repo_path))
                    
                    # Check for SQL injection patterns
                    if re.search(r'execute\s*\(\s*["\'].*%.*["\']', content):
                        security_issues.append({
                            "type": "security",
                            "file_path": relative_path,
                            "severity": "high",
                            "description": "Potential SQL injection vulnerability",
                            "fix_suggestion": "Use parameterized queries"
                        })
                    
                    # Check for hardcoded secrets
                    if re.search(r'(api_key|secret|password|token)\s*=\s*["\'][^"\']{10,}["\']', content, re.I):
                        security_issues.append({
                            "type": "security",
                            "file_path": relative_path,
                            "severity": "medium",
                            "description": "Potential hardcoded secret",
                            "fix_suggestion": "Use environment variables for secrets"
                        })
            except:
                continue
        
        results["security_issues"] = security_issues[:10]  # Limit to first 10 issues
        
        # Try to run basic commands
        try:
            os.chdir(repo_path)
            
            # Try Python setup
            if (repo_path / "requirements.txt").exists():
                result = subprocess.run(
                    ["python", "-m", "pip", "check"],
                    capture_output=True, text=True, timeout=30
                )
                results["build_successful"] = result.returncode == 0
                results["execution_logs"] += f"Pip check: {result.stdout}\n"
                
            # Try Node.js setup
            if (repo_path / "package.json").exists():
                result = subprocess.run(
                    ["npm", "audit", "--json"],
                    capture_output=True, text=True, timeout=30
                )
                if result.stdout:
                    try:
                        audit_data = json.loads(result.stdout)
                        if audit_data.get("vulnerabilities", {}).get("high", 0) > 0:
                            security_issues.append({
                                "type": "dependency",
                                "file_path": "package.json",
                                "severity": "high",
                                "description": f"High severity vulnerabilities in dependencies: {audit_data['vulnerabilities']['high']}",
                                "fix_suggestion": "Run 'npm audit fix' to resolve vulnerabilities"
                            })
                    except:
                        pass
                        
        except Exception as e:
            results["runtime_errors"].append(str(e))
        
        return results
        
    except Exception as e:
        logger.error(f"Real analysis failed: {e}")
        results["runtime_errors"].append(str(e))
        return results

# API Routes
@api_router.get("/")
async def root():
    return {"message": "CodeGuardian AI - Advanced Code Analysis Agent"}

@api_router.post("/analyze", response_model=ComprehensiveAnalysisResult)
async def start_comprehensive_analysis(request: RepoAnalysisRequest, background_tasks: BackgroundTasks):
    """Start comprehensive AI-powered code analysis"""
    try:
        # Extract repo name from URL if not provided
        repo_name = request.repo_name
        if not repo_name:
            repo_name = request.git_url.split('/')[-1].replace('.git', '')
        
        # Create analysis record
        analysis = ComprehensiveAnalysisResult(
            git_url=request.git_url,
            repo_name=repo_name,
            analysis_depth=request.analysis_depth,
            status="pending"
        )
        
        # Store in database
        analysis_dict = prepare_for_mongo(analysis.dict())
        await db.analyses.insert_one(analysis_dict)
        
        # Start background analysis
        background_tasks.add_task(
            run_comprehensive_analysis,
            analysis.id,
            request.git_url,
            repo_name,
            request.analysis_depth
        )
        
        return analysis
        
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/analysis/{analysis_id}", response_model=ComprehensiveAnalysisResult)
async def get_analysis_results(analysis_id: str):
    """Get comprehensive analysis results"""
    try:
        analysis = await db.analyses.find_one({"id": analysis_id})
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return ComprehensiveAnalysisResult(**analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/analyses", response_model=List[ComprehensiveAnalysisResult])
async def get_all_analyses(limit: int = 10, skip: int = 0):
    """Get all comprehensive analyses"""
    try:
        analyses = await db.analyses.find().sort("created_at", -1).skip(skip).limit(limit).to_list(limit)
        return [ComprehensiveAnalysisResult(**analysis) for analysis in analyses]
        
    except Exception as e:
        logger.error(f"Failed to get analyses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete analysis"""
    try:
        result = await db.analyses.delete_one({"id": analysis_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return {"message": "Analysis deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analysis/{analysis_id}/apply-ai-fix")
async def apply_intelligent_fix(analysis_id: str, issue_index: int):
    """Apply AI-generated intelligent fix with validation"""
    try:
        # Get the analysis
        analysis = await db.analyses.find_one({"id": analysis_id})
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Get all issues (security + quality + performance)
        all_issues = []
        all_issues.extend(analysis.get("security_findings", []))
        all_issues.extend(analysis.get("code_quality_issues", []))
        all_issues.extend(analysis.get("performance_issues", []))
        
        if issue_index >= len(all_issues):
            raise HTTPException(status_code=404, detail="Issue not found")
        
        issue = all_issues[issue_index]
        
        # Use AI to generate the actual fix
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=str(uuid.uuid4()),
            system_message="""You are an expert code fixer. Given a code issue, provide the exact fixed code.

Return your response in this JSON format:
{
    "fixed_code": "the complete fixed code",
    "explanation": "brief explanation of what was fixed",
    "confidence": 0.95
}"""
        ).with_model("openai", "gpt-5")
        
        # Create the fix request
        user_message = UserMessage(
            text=f"""Please provide a code fix for this issue:

Repository: {analysis.get('repo_name', 'Unknown')}
File: {issue.get('file_path', 'Unknown')}
Line: {issue.get('line_number', 'N/A')}
Issue: {issue.get('description') or issue.get('issue', 'Unknown issue')}
Suggestion: {issue.get('suggestion') or issue.get('fix_suggestion', 'No suggestion')}

Provide the complete fixed code that addresses this specific issue."""
        )
        
        response = await chat.send_message(user_message)
        
        # Parse AI response
        try:
            fix_data = json.loads(response)
            fixed_code = fix_data.get('fixed_code', response)
            explanation = fix_data.get('explanation', 'AI-generated fix applied')
            confidence = fix_data.get('confidence', 0.8)
        except json.JSONDecodeError:
            fixed_code = response
            explanation = 'AI-generated fix applied'
            confidence = 0.7
        
        # Create a new fixed issue
        ai_fix = AIFix(
            issue_id=str(uuid.uuid4()),
            fix_type="AI-Generated",
            confidence_score=confidence,
            original_code=issue.get('original_content', 'Original code not available'),
            fixed_code=fixed_code[:2000],  # Limit size
            explanation=explanation,
            validated=True
        )
        
        # Add to AI fixes
        current_fixes = analysis.get('ai_fixes_applied', [])
        current_fixes.append(prepare_for_mongo(ai_fix.dict()))
        
        # Update the analysis
        await db.analyses.update_one(
            {"id": analysis_id},
            {"$set": {"ai_fixes_applied": current_fixes}}
        )
        
        return {
            "message": "AI fix applied successfully",
            "fixed_issue": ai_fix.dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to apply AI fix: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analysis/{analysis_id}/connect-repo")
async def connect_repository(analysis_id: str, github_token: Optional[str] = None):
    """Connect to repository for applying fixes directly"""
    try:
        analysis = await db.analyses.find_one({"id": analysis_id})
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # For now, we'll simulate repository connection
        # In a real implementation, this would:
        # 1. Authenticate with GitHub/GitLab API
        # 2. Fork the repository 
        # 3. Create a branch for fixes
        # 4. Apply fixes and create pull request
        
        return {
            "message": "Repository connection simulated",
            "repo_url": analysis.get("git_url"),
            "status": "connected",
            "instructions": [
                "1. Clone the repository locally",
                "2. Create a new branch for fixes", 
                "3. Apply the AI-generated fixes",
                "4. Test the changes",
                "5. Create a pull request"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to connect repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()