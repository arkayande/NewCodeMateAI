import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Badge } from './components/ui/badge';
import { Separator } from './components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Progress } from './components/ui/progress';
import { 
  AlertCircle, 
  CheckCircle2, 
  Clock, 
  GitBranch, 
  Shield, 
  Bug, 
  Code, 
  Zap, 
  Trash2, 
  X,
  Brain,
  Server,
  Target,
  TrendingUp,
  Lock,
  Cpu,
  Database,
  TestTube,
  Layers,
  Activity,
  Wand2,
  GitPullRequest
} from 'lucide-react';
import { useToast } from './hooks/use-toast';
import { Toaster } from './components/ui/toaster';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Home = () => {
  const [gitUrl, setGitUrl] = useState('');
  const [analysisDepth, setAnalysisDepth] = useState('comprehensive');
  const [analyses, setAnalyses] = useState([]);
  const [currentAnalysis, setCurrentAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [fixingIssues, setFixingIssues] = useState(new Set()); // Track which issues are being fixed
  const { toast } = useToast();

  const applyAiFix = async (analysisId, issueIndex) => {
    const fixKey = `${analysisId}-${issueIndex}`;
    setFixingIssues(prev => new Set([...prev, fixKey]));
    
    try {
      const response = await axios.post(`${API}/analysis/${analysisId}/apply-ai-fix?issue_index=${issueIndex}`);
      
      toast({
        title: 'AI Fix Applied',
        description: 'The issue has been automatically fixed by AI',
      });
      
      // Refresh the current analysis to show the new fix
      const updatedAnalysis = await axios.get(`${API}/analysis/${analysisId}`);
      setCurrentAnalysis(updatedAnalysis.data);
      
      // Also refresh the analyses list
      fetchAnalyses();
      
    } catch (error) {
      console.error('Failed to apply AI fix:', error);
      toast({
        title: 'Fix Failed',
        description: 'Could not apply the AI fix. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setFixingIssues(prev => {
        const newSet = new Set(prev);
        newSet.delete(fixKey);
        return newSet;
      });
    }
  };

  const connectRepository = async (analysisId) => {
    try {
      const response = await axios.post(`${API}/analysis/${analysisId}/connect-repo`);
      
      toast({
        title: 'Repository Connection',
        description: 'Repository connection instructions provided',
      });
      
      // You could show a modal with the instructions here
      console.log('Repository connection instructions:', response.data.instructions);
      
    } catch (error) {
      console.error('Failed to connect repository:', error);
      toast({
        title: 'Connection Failed',
        description: 'Could not connect to repository. Please try again.',
        variant: 'destructive',
      });
    }
  };

  const fetchAnalyses = async () => {
    try {
      const response = await axios.get(`${API}/analyses`);
      setAnalyses(response.data);
    } catch (error) {
      console.error('Failed to fetch analyses:', error);
    }
  };

  const deleteAnalysis = async (analysisId, repoName) => {
    try {
      await axios.delete(`${API}/analysis/${analysisId}`);
      
      setAnalyses(prev => prev.filter(analysis => analysis.id !== analysisId));
      
      if (currentAnalysis && currentAnalysis.id === analysisId) {
        setCurrentAnalysis(null);
      }
      
      toast({
        title: 'Analysis Deleted',
        description: `Successfully deleted analysis for ${repoName}`,
      });
    } catch (error) {
      console.error('Failed to delete analysis:', error);
      toast({
        title: 'Error',
        description: 'Failed to delete analysis. Please try again.',
        variant: 'destructive',
      });
    }
  };

  const startAnalysis = async () => {
    if (!gitUrl.trim()) {
      toast({
        title: 'Error',
        description: 'Please enter a Git repository URL',
        variant: 'destructive',
      });
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API}/analyze`, {
        git_url: gitUrl.trim(),
        analysis_depth: analysisDepth
      });
      
      setCurrentAnalysis(response.data);
      setGitUrl('');
      fetchAnalyses();
      
      toast({
        title: 'Advanced Analysis Started',
        description: 'AI-powered comprehensive analysis initiated. This may take several minutes.',
      });

      // Poll for updates
      pollAnalysis(response.data.id);
    } catch (error) {
      console.error('Failed to start analysis:', error);
      toast({
        title: 'Error',
        description: 'Failed to start repository analysis. Please check the URL and try again.',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  const pollAnalysis = async (analysisId) => {
    const maxAttempts = 60; // 10 minutes max
    let attempts = 0;

    const poll = async () => {
      try {
        const response = await axios.get(`${API}/analysis/${analysisId}`);
        const analysis = response.data;
        
        setCurrentAnalysis(analysis);
        
        if (analysis.status === 'completed' || analysis.status === 'failed') {
          fetchAnalyses();
          return;
        }
        
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 10000); // Poll every 10 seconds
        }
      } catch (error) {
        console.error('Failed to poll analysis:', error);
      }
    };

    poll();
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case 'analyzing': return <Brain className="h-4 w-4 text-blue-500 animate-pulse" />;
      case 'failed': return <AlertCircle className="h-4 w-4 text-red-500" />;
      default: return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getRatingColor = (rating) => {
    switch (rating?.toUpperCase()) {
      case 'A': return 'text-green-600';
      case 'B': return 'text-blue-600';
      case 'C': return 'text-yellow-600';
      case 'D': return 'text-orange-600';
      case 'F': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getDeploymentReadinessColor = (readiness) => {
    switch (readiness) {
      case 'ready': return 'text-green-600 bg-green-100';
      case 'needs_work': return 'text-yellow-600 bg-yellow-100';
      case 'not_ready': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  useEffect(() => {
    fetchAnalyses();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-purple-600 to-blue-600 rounded-2xl shadow-lg">
              <Brain className="h-8 w-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
              CodeGuardian AI Agent
            </h1>
          </div>
          <p className="text-lg text-slate-600 max-w-3xl mx-auto">
            Advanced AI-powered code analysis agent with Docker sandbox execution, deep security scanning, 
            performance analysis, and intelligent auto-fixing capabilities
          </p>
        </div>

        {/* Analysis Input */}
        <Card className="mb-8 shadow-lg border-0 bg-white/80 backdrop-blur-sm" data-testid="analysis-input-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-5 w-5" />
              Advanced Repository Analysis
            </CardTitle>
            <CardDescription>
              Enter a Git repository URL for comprehensive AI-powered analysis with Docker sandbox execution
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-4">
              <Input
                placeholder="https://github.com/username/repository.git"
                value={gitUrl}
                onChange={(e) => setGitUrl(e.target.value)}
                className="flex-1"
                onKeyPress={(e) => e.key === 'Enter' && startAnalysis()}
                data-testid="git-url-input"
              />
              <Select value={analysisDepth} onValueChange={setAnalysisDepth}>
                <SelectTrigger className="w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="basic">Basic Analysis</SelectItem>
                  <SelectItem value="standard">Standard Analysis</SelectItem>
                  <SelectItem value="comprehensive">Comprehensive Analysis</SelectItem>
                </SelectContent>
              </Select>
              <Button 
                onClick={startAnalysis} 
                disabled={loading}
                className="px-8 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
                data-testid="analyze-button"
              >
                {loading ? (
                  <>
                    <Brain className="h-4 w-4 mr-2 animate-pulse" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Zap className="h-4 w-4 mr-2" />
                    Start AI Analysis
                  </>
                )}
              </Button>
            </div>
            
            {/* Analysis Depth Description */}
            <div className="text-sm text-slate-500 bg-slate-50 p-3 rounded-lg">
              <strong>Analysis Depths:</strong> 
              <ul className="mt-1 space-y-1">
                <li><strong>Basic:</strong> Code scanning, basic security checks</li>
                <li><strong>Standard:</strong> + Dependencies, build testing, performance analysis</li>
                <li><strong>Comprehensive:</strong> + AI architecture analysis, execution testing, advanced security</li>
              </ul>
            </div>
          </CardContent>
        </Card>

        {/* Current Analysis */}
        {currentAnalysis && (
          <Card className="mb-8 shadow-lg border-0 bg-white/80 backdrop-blur-sm" data-testid="current-analysis">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {getStatusIcon(currentAnalysis.status)}
                  Current Analysis: {currentAnalysis.repo_name}
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="capitalize">
                    {currentAnalysis.status}
                  </Badge>
                  <Badge variant="outline" className="capitalize">
                    {currentAnalysis.analysis_depth}
                  </Badge>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setCurrentAnalysis(null)}
                    className="h-6 w-6 p-0"
                    data-testid="close-current-analysis"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </CardTitle>
              <CardDescription>
                {currentAnalysis.git_url}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {currentAnalysis.status === 'completed' ? (
                <Tabs defaultValue="overview" className="w-full">
                  <TabsList className="grid w-full grid-cols-7">
                    <TabsTrigger value="overview">Overview</TabsTrigger>
                    <TabsTrigger value="security">Security</TabsTrigger>
                    <TabsTrigger value="performance">Performance</TabsTrigger>
                    <TabsTrigger value="quality">Quality</TabsTrigger>
                    <TabsTrigger value="ai-fixes">AI Fixes</TabsTrigger>
                    <TabsTrigger value="architecture">Architecture</TabsTrigger>
                    <TabsTrigger value="execution">Execution</TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="overview" className="space-y-6">
                    {/* Key Metrics */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center p-4 bg-slate-50 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">{currentAnalysis.total_files || 0}</div>
                        <div className="text-sm text-slate-600">Files Analyzed</div>
                      </div>
                      <div className="text-center p-4 bg-slate-50 rounded-lg">
                        <div className="text-2xl font-bold text-purple-600">{currentAnalysis.lines_of_code || 0}</div>
                        <div className="text-sm text-slate-600">Lines of Code</div>
                      </div>
                      <div className="text-center p-4 bg-slate-50 rounded-lg">
                        <div className="text-2xl font-bold text-orange-600">{currentAnalysis.security_findings?.length || 0}</div>
                        <div className="text-sm text-slate-600">Security Issues</div>
                      </div>
                      <div className="text-center p-4 bg-slate-50 rounded-lg">
                        <div className="text-2xl font-bold text-green-600">{currentAnalysis.ai_fixes_applied?.length || 0}</div>
                        <div className="text-sm text-slate-600">AI Fixes Applied</div>
                      </div>
                    </div>

                    {/* AI Ratings */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <Card>
                        <CardContent className="pt-6">
                          <div className="flex items-center gap-3">
                            <Shield className="h-8 w-8 text-red-500" />
                            <div>
                              <div className="text-sm text-slate-600">Security Rating</div>
                              <div className={`text-2xl font-bold ${getRatingColor(currentAnalysis.security_rating)}`}>
                                {currentAnalysis.security_rating || 'N/A'}
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                      
                      <Card>
                        <CardContent className="pt-6">
                          <div className="flex items-center gap-3">
                            <Code className="h-8 w-8 text-blue-500" />
                            <div>
                              <div className="text-sm text-slate-600">Code Quality Rating</div>
                              <div className={`text-2xl font-bold ${getRatingColor(currentAnalysis.code_quality_rating)}`}>
                                {currentAnalysis.code_quality_rating || 'N/A'}
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                      
                      <Card>
                        <CardContent className="pt-6">
                          <div className="flex items-center gap-3">
                            <TrendingUp className="h-8 w-8 text-green-500" />
                            <div>
                              <div className="text-sm text-slate-600">Performance Rating</div>
                              <div className={`text-2xl font-bold ${getRatingColor(currentAnalysis.performance_rating)}`}>
                                {currentAnalysis.performance_rating || 'N/A'}
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>

                    {/* Technologies & Deployment Readiness */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-lg flex items-center gap-2">
                            <Layers className="h-5 w-5" />
                            Technologies Detected
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-2">
                            <div>
                              <span className="font-medium">Languages: </span>
                              {currentAnalysis.languages_detected?.join(', ') || 'None detected'}
                            </div>
                            <div>
                              <span className="font-medium">Frameworks: </span>
                              {currentAnalysis.framework_detected?.join(', ') || 'None detected'}
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      <Card>
                        <CardHeader>
                          <CardTitle className="text-lg flex items-center gap-2">
                            <Target className="h-5 w-5" />
                            Deployment Readiness
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <Badge className={`text-sm px-3 py-1 ${getDeploymentReadinessColor(currentAnalysis.deployment_readiness)}`}>
                            {currentAnalysis.deployment_readiness || 'Unknown'}
                          </Badge>
                          <div className="mt-2 text-sm text-slate-600">
                            Analysis Duration: {currentAnalysis.analysis_duration ? `${currentAnalysis.analysis_duration.toFixed(1)}s` : 'N/A'}
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  </TabsContent>
                  
                  <TabsContent value="security" className="space-y-4">
                    {currentAnalysis.security_findings && currentAnalysis.security_findings.length > 0 ? (
                      currentAnalysis.security_findings.map((finding, idx) => {
                        const globalIndex = idx; // Security findings start at index 0
                        const fixKey = `${currentAnalysis.id}-${globalIndex}`;
                        const isFixing = fixingIssues.has(fixKey);
                        
                        return (
                          <Card key={idx} className="border-l-4 border-l-red-500">
                            <CardContent className="pt-6">
                              <div className="flex items-start justify-between">
                                <div className="flex-1">
                                  <div className="font-medium text-slate-800">{finding.file_path}</div>
                                  {finding.line_number && (
                                    <div className="text-sm text-slate-500 mb-2">Line {finding.line_number}</div>
                                  )}
                                  <p className="text-sm text-slate-700 mb-2">{finding.description}</p>
                                  {finding.fix_suggestion && (
                                    <p className="text-sm text-blue-600 mb-3">{finding.fix_suggestion}</p>
                                  )}
                                  
                                  {/* AI Auto-Fix Button */}
                                  <div className="flex gap-2">
                                    <Button
                                      size="sm"
                                      onClick={() => applyAiFix(currentAnalysis.id, globalIndex)}
                                      disabled={isFixing}
                                      className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white"
                                      data-testid={`ai-fix-security-${globalIndex}`}
                                    >
                                      {isFixing ? (
                                        <>
                                          <Clock className="h-4 w-4 mr-2 animate-spin" />
                                          Fixing...
                                        </>
                                      ) : (
                                        <>
                                          <Wand2 className="h-4 w-4 mr-2" />
                                          AI Auto-Fix
                                        </>
                                      )}
                                    </Button>
                                  </div>
                                </div>
                                <Badge className={`ml-2 ${finding.severity === 'critical' ? 'bg-red-600' : finding.severity === 'high' ? 'bg-orange-500' : 'bg-yellow-500'} text-white`}>
                                  {finding.severity}
                                </Badge>
                              </div>
                            </CardContent>
                          </Card>
                        );
                      })
                    ) : (
                      <div className="text-center py-8 text-slate-500">
                        <Shield className="h-12 w-12 mx-auto mb-4 text-green-500" />
                        <p>No security vulnerabilities detected!</p>
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="performance" className="space-y-4">
                    {currentAnalysis.performance_issues && currentAnalysis.performance_issues.length > 0 ? (
                      currentAnalysis.performance_issues.map((issue, idx) => (
                        <Card key={idx} className="border-l-4 border-l-yellow-500">
                          <CardContent className="pt-6">
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <div className="font-medium text-slate-800">{issue.file_path}</div>
                                {issue.function_name && (
                                  <div className="text-sm text-slate-500 mb-2">Function: {issue.function_name}</div>
                                )}
                                <p className="text-sm text-slate-700 mb-2">{issue.issue}</p>
                                <p className="text-sm text-blue-600">{issue.optimization_suggestion}</p>
                                {issue.estimated_improvement && (
                                  <div className="text-sm text-green-600 mt-1">
                                    Estimated improvement: {issue.estimated_improvement}
                                  </div>
                                )}
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))
                    ) : (
                      <div className="text-center py-8 text-slate-500">
                        <TrendingUp className="h-12 w-12 mx-auto mb-4 text-green-500" />
                        <p>No performance issues detected!</p>
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="quality" className="space-y-4">
                    {currentAnalysis.code_quality_issues && currentAnalysis.code_quality_issues.length > 0 ? (
                      currentAnalysis.code_quality_issues.map((issue, idx) => {
                        const globalIndex = (currentAnalysis.security_findings?.length || 0) + idx; // Quality issues after security
                        const fixKey = `${currentAnalysis.id}-${globalIndex}`;
                        const isFixing = fixingIssues.has(fixKey);
                        
                        return (
                          <Card key={idx} className="border-l-4 border-l-blue-500">
                            <CardContent className="pt-6">
                              <div className="flex items-start justify-between">
                                <div className="flex-1">
                                  <div className="font-medium text-slate-800">{issue.file_path}</div>
                                  {issue.line_number && (
                                    <div className="text-sm text-slate-500 mb-2">Line {issue.line_number}</div>
                                  )}
                                  <p className="text-sm text-slate-700 mb-2">{issue.issue}</p>
                                  <p className="text-sm text-blue-600 mb-3">{issue.suggestion}</p>
                                  
                                  {/* AI Auto-Fix Button for auto-fixable issues */}
                                  {issue.auto_fixable && (
                                    <div className="flex gap-2">
                                      <Button
                                        size="sm"
                                        onClick={() => applyAiFix(currentAnalysis.id, globalIndex)}
                                        disabled={isFixing}
                                        className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white"
                                        data-testid={`ai-fix-quality-${globalIndex}`}
                                      >
                                        {isFixing ? (
                                          <>
                                            <Clock className="h-4 w-4 mr-2 animate-spin" />
                                            Fixing...
                                          </>
                                        ) : (
                                          <>
                                            <Wand2 className="h-4 w-4 mr-2" />
                                            AI Auto-Fix
                                          </>
                                        )}
                                      </Button>
                                    </div>
                                  )}
                                </div>
                                <div className="flex items-start gap-2 ml-4">
                                  <Badge className={`${issue.severity === 'high' ? 'bg-orange-500' : 'bg-blue-500'} text-white`}>
                                    {issue.severity}
                                  </Badge>
                                  {issue.auto_fixable && (
                                    <Badge variant="outline" className="text-green-600 border-green-600">
                                      Auto-fixable
                                    </Badge>
                                  )}
                                </div>
                              </div>
                            </CardContent>
                          </Card>
                        );
                      })
                    ) : (
                      <div className="text-center py-8 text-slate-500">
                        <Code className="h-12 w-12 mx-auto mb-4 text-green-500" />
                        <p>No code quality issues detected!</p>
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="architecture" className="space-y-4">
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Brain className="h-5 w-5" />
                          AI Architecture Analysis
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-slate-700 leading-relaxed whitespace-pre-wrap">
                          {currentAnalysis.architecture_analysis || 'AI architecture analysis is not available.'}
                        </p>
                      </CardContent>
                    </Card>

                    {currentAnalysis.recommendations && currentAnalysis.recommendations.length > 0 && (
                      <Card>
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2">
                            <Target className="h-5 w-5" />
                            AI Recommendations
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <ul className="space-y-2">
                            {currentAnalysis.recommendations.map((rec, idx) => (
                              <li key={idx} className="flex items-start gap-2">
                                <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                                <span className="text-sm text-slate-700">{rec}</span>
                              </li>
                            ))}
                          </ul>
                        </CardContent>
                      </Card>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="execution" className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <Card>
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2">
                            <Activity className="h-5 w-5" />
                            Build Results
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="flex items-center gap-2">
                            {currentAnalysis.build_successful ? (
                              <CheckCircle2 className="h-5 w-5 text-green-500" />
                            ) : (
                              <AlertCircle className="h-5 w-5 text-red-500" />
                            )}
                            <span className="font-medium">
                              {currentAnalysis.build_successful ? 'Build Successful' : 'Build Failed'}
                            </span>
                          </div>
                        </CardContent>
                      </Card>

                      <Card>
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2">
                            <TestTube className="h-5 w-5" />
                            Test Execution
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="text-sm text-slate-600">
                            {currentAnalysis.test_results?.length > 0 ? (
                              <div>Tests executed: {currentAnalysis.test_results.length}</div>
                            ) : (
                              <div>No tests executed</div>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    </div>

                    {currentAnalysis.execution_logs && (
                      <Card>
                        <CardHeader>
                          <CardTitle>Execution Logs</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <pre className="text-xs bg-slate-900 text-green-400 p-4 rounded overflow-auto max-h-60">
                            {currentAnalysis.execution_logs}
                          </pre>
                        </CardContent>
                      </Card>
                    )}
                  </TabsContent>
                </Tabs>
              ) : currentAnalysis.status === 'analyzing' ? (
                <div className="text-center py-8">
                  <Brain className="h-12 w-12 mx-auto mb-4 text-purple-500 animate-pulse" />
                  <p className="text-slate-600 mb-2">AI Agent is analyzing your repository...</p>
                  <p className="text-sm text-slate-500">
                    Docker sandbox execution in progress. This may take several minutes.
                  </p>
                </div>
              ) : (
                <div className="text-center py-8">
                  <Clock className="h-12 w-12 mx-auto mb-4 text-blue-500" />
                  <p className="text-slate-600">Initializing analysis...</p>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Analysis History */}
        <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
          <CardHeader>
            <CardTitle>Analysis History</CardTitle>
            <CardDescription>Previous comprehensive repository analyses</CardDescription>
          </CardHeader>
          <CardContent>
            {analyses.length > 0 ? (
              <div className="space-y-4">
                {analyses.map((analysis) => (
                  <div key={analysis.id} className="p-4 bg-slate-50 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          {getStatusIcon(analysis.status)}
                          <span className="font-medium">{analysis.repo_name}</span>
                          <Badge variant="secondary" className="capitalize text-xs">
                            {analysis.status}
                          </Badge>
                          <Badge variant="outline" className="capitalize text-xs">
                            {analysis.analysis_depth}
                          </Badge>
                        </div>
                        <div className="text-sm text-slate-500 mb-2">{analysis.git_url}</div>
                        {analysis.status === 'completed' && (
                          <div className="flex gap-4 text-sm">
                            <span className="text-slate-600">
                              {analysis.total_files} files
                            </span>
                            <span className="text-red-600">
                              {analysis.security_findings?.length || 0} security
                            </span>
                            <span className="text-blue-600">
                              {analysis.code_quality_issues?.length || 0} quality
                            </span>
                            <span className="text-green-600">
                              {analysis.ai_fixes_applied?.length || 0} fixes
                            </span>
                            {analysis.deployment_readiness && (
                              <Badge className={`text-xs ${getDeploymentReadinessColor(analysis.deployment_readiness)}`}>
                                {analysis.deployment_readiness}
                              </Badge>
                            )}
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setCurrentAnalysis(analysis)}
                          data-testid={`view-analysis-${analysis.id}`}
                        >
                          View
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => deleteAnalysis(analysis.id, analysis.repo_name)}
                          className="text-red-600 hover:text-red-700 hover:bg-red-50"
                          data-testid={`delete-analysis-${analysis.id}`}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-slate-500">
                <Brain className="h-12 w-12 mx-auto mb-4" />
                <p>No analyses yet. Start by analyzing your first repository!</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
      <Toaster />
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;