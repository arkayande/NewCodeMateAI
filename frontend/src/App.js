import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Badge } from './components/ui/badge';
import { Separator } from './components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { AlertCircle, CheckCircle2, Clock, GitBranch, Shield, Bug, Code, Zap, Trash2, X } from 'lucide-react';
import { useToast } from './hooks/use-toast';
import { Toaster } from './components/ui/toaster';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Home = () => {
  const [gitUrl, setGitUrl] = useState('');
  const [analyses, setAnalyses] = useState([]);
  const [currentAnalysis, setCurrentAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  const deleteAnalysis = async (analysisId, repoName) => {
    try {
      await axios.delete(`${API}/analysis/${analysisId}`);
      
      // Remove from local state
      setAnalyses(prev => prev.filter(analysis => analysis.id !== analysisId));
      
      // Clear current analysis if it was the deleted one
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

  const fetchAnalyses = async () => {
    try {
      const response = await axios.get(`${API}/analyses`);
      setAnalyses(response.data);
    } catch (error) {
      console.error('Failed to fetch analyses:', error);
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
      });
      
      setCurrentAnalysis(response.data);
      setGitUrl('');
      fetchAnalyses();
      
      toast({
        title: 'Analysis Started',
        description: 'Repository analysis has been initiated. Results will appear shortly.',
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
    const maxAttempts = 30; // 5 minutes max
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

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical': return 'bg-red-500';
      case 'high': return 'bg-orange-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case 'analyzing': return <Clock className="h-4 w-4 text-blue-500" />;
      case 'failed': return <AlertCircle className="h-4 w-4 text-red-500" />;
      default: return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const groupIssuesByType = (issues) => {
    return issues.reduce((acc, issue) => {
      if (!acc[issue.error_type]) {
        acc[issue.error_type] = [];
      }
      acc[issue.error_type].push(issue);
      return acc;
    }, {});
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
            <div className="p-3 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-2xl shadow-lg">
              <Shield className="h-8 w-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              CodeGuardian AI
            </h1>
          </div>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            Advanced AI-powered code analysis that detects conflicts, vulnerabilities, and deployment issues automatically
          </p>
        </div>

        {/* Analysis Input */}
        <Card className="mb-8 shadow-lg border-0 bg-white/80 backdrop-blur-sm" data-testid="analysis-input-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <GitBranch className="h-5 w-5" />
              Analyze Repository
            </CardTitle>
            <CardDescription>
              Enter a Git repository URL to start comprehensive code analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex gap-4">
              <Input
                placeholder="https://github.com/username/repository.git"
                value={gitUrl}
                onChange={(e) => setGitUrl(e.target.value)}
                className="flex-1"
                onKeyPress={(e) => e.key === 'Enter' && startAnalysis()}
                data-testid="git-url-input"
              />
              <Button 
                onClick={startAnalysis} 
                disabled={loading}
                className="px-8 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
                data-testid="analyze-button"
              >
                {loading ? (
                  <>
                    <Clock className="h-4 w-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Zap className="h-4 w-4 mr-2" />
                    Analyze
                  </>
                )}
              </Button>
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
                  <TabsList className="grid w-full grid-cols-4">
                    <TabsTrigger value="overview">Overview</TabsTrigger>
                    <TabsTrigger value="issues">Issues</TabsTrigger>
                    <TabsTrigger value="fixes">Auto-Fixes</TabsTrigger>
                    <TabsTrigger value="summary">AI Summary</TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="overview" className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="text-center p-4 bg-slate-50 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">{currentAnalysis.total_files_analyzed}</div>
                        <div className="text-sm text-slate-600">Files Analyzed</div>
                      </div>
                      <div className="text-center p-4 bg-slate-50 rounded-lg">
                        <div className="text-2xl font-bold text-orange-600">{currentAnalysis.issues_found?.length || 0}</div>
                        <div className="text-sm text-slate-600">Issues Found</div>
                      </div>
                      <div className="text-center p-4 bg-slate-50 rounded-lg">
                        <div className="text-2xl font-bold text-green-600">{currentAnalysis.fixes_applied?.length || 0}</div>
                        <div className="text-sm text-slate-600">Auto-Fixes Applied</div>
                      </div>
                    </div>
                  </TabsContent>
                  
                  <TabsContent value="issues" className="space-y-4">
                    {currentAnalysis.issues_found && currentAnalysis.issues_found.length > 0 ? (
                      Object.entries(groupIssuesByType(currentAnalysis.issues_found)).map(([type, issues]) => (
                        <Card key={type} className="border-l-4 border-l-orange-500">
                          <CardHeader>
                            <CardTitle className="flex items-center gap-2 text-lg">
                              <Bug className="h-5 w-5" />
                              {type} ({issues.length})
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="space-y-3">
                            {issues.map((issue, idx) => (
                              <div key={idx} className="p-3 bg-slate-50 rounded-lg">
                                <div className="flex items-start justify-between">
                                  <div className="flex-1">
                                    <div className="font-medium text-slate-800">{issue.file_path}</div>
                                    {issue.line_number && (
                                      <div className="text-sm text-slate-500 mb-2">Line {issue.line_number}</div>
                                    )}
                                    <p className="text-sm text-slate-700 mb-2">{issue.description}</p>
                                    <p className="text-sm text-blue-600">{issue.suggestion}</p>
                                  </div>
                                  <Badge className={`ml-2 ${getSeverityColor(issue.severity)} text-white`}>
                                    {issue.severity}
                                  </Badge>
                                </div>
                              </div>
                            ))}
                          </CardContent>
                        </Card>
                      ))
                    ) : (
                      <div className="text-center py-8 text-slate-500">
                        <CheckCircle2 className="h-12 w-12 mx-auto mb-4 text-green-500" />
                        <p>No issues found! Your code looks great.</p>
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="fixes" className="space-y-4">
                    {currentAnalysis.fixes_applied && currentAnalysis.fixes_applied.length > 0 ? (
                      currentAnalysis.fixes_applied.map((fix, idx) => (
                        <Card key={idx} className="border-l-4 border-l-green-500">
                          <CardContent className="pt-6">
                            <div className="flex items-start justify-between mb-4">
                              <div className="flex-1">
                                <div className="font-medium text-slate-800 mb-1">{fix.file_path}</div>
                                {fix.line_number && (
                                  <div className="text-sm text-slate-500 mb-2">Line {fix.line_number}</div>
                                )}
                                <p className="text-sm text-slate-700 mb-2">{fix.description}</p>
                                <p className="text-sm text-green-600 mb-3">✓ {fix.suggestion}</p>
                              </div>
                              <Badge className="ml-2 bg-green-500 text-white">Fixed</Badge>
                            </div>
                            
                            {fix.original_content && fix.fixed_content && (
                              <div className="space-y-3">
                                <Separator />
                                <div>
                                  <div className="text-sm font-medium text-red-600 mb-2">Before:</div>
                                  <div className="bg-red-50 border border-red-200 rounded p-3">
                                    <code className="text-sm text-red-800 whitespace-pre-wrap">
                                      {fix.original_content}
                                    </code>
                                  </div>
                                </div>
                                <div>
                                  <div className="text-sm font-medium text-green-600 mb-2">After:</div>
                                  <div className="bg-green-50 border border-green-200 rounded p-3">
                                    <code className="text-sm text-green-800 whitespace-pre-wrap">
                                      {fix.fixed_content}
                                    </code>
                                  </div>
                                </div>
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      ))
                    ) : (
                      <div className="text-center py-8 text-slate-500">
                        <Code className="h-12 w-12 mx-auto mb-4" />
                        <p>No automatic fixes were applied.</p>
                        <p className="text-sm mt-2">This repository had no auto-fixable issues detected.</p>
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="summary" className="space-y-4">
                    <Card>
                      <CardContent className="pt-6">
                        <p className="text-slate-700 leading-relaxed">
                          {currentAnalysis.ai_summary || 'AI analysis summary is not available.'}
                        </p>
                      </CardContent>
                    </Card>
                  </TabsContent>
                </Tabs>
              ) : (
                <div className="text-center py-8">
                  <Clock className="h-12 w-12 mx-auto mb-4 text-blue-500 animate-pulse" />
                  <p className="text-slate-600">
                    {currentAnalysis.status === 'analyzing' ? 'Analyzing your repository...' : 'Starting analysis...'}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Analysis History */}
        <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
          <CardHeader>
            <CardTitle>Analysis History</CardTitle>
            <CardDescription>Previous repository analyses</CardDescription>
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
                        </div>
                        <div className="text-sm text-slate-500 mb-2">{analysis.git_url}</div>
                        {analysis.status === 'completed' && (
                          <div className="flex gap-4 text-sm">
                            <span className="text-slate-600">
                              {analysis.total_files_analyzed} files analyzed
                            </span>
                            <span className="text-orange-600">
                              {analysis.issues_found?.length || 0} issues
                            </span>
                            <span className="text-green-600">
                              {analysis.fixes_applied?.length || 0} fixes
                            </span>
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
                <GitBranch className="h-12 w-12 mx-auto mb-4" />
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