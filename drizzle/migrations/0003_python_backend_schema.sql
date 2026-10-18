CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS public.chat_conversations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id uuid NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  title text NOT NULL DEFAULT 'Beszélgetés',
  messages jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.chat_jobs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id uuid NOT NULL REFERENCES public.chat_conversations(id) ON DELETE CASCADE,
  history jsonb NOT NULL,
  status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'cancelled', 'failed')),
  response_content text NOT NULL DEFAULT '',
  response_reasoning text NOT NULL DEFAULT '',
  error text NULL,
  stop_requested boolean NOT NULL DEFAULT false,
  lease_expires_at timestamptz NULL,
  started_at timestamptz NULL,
  finished_at timestamptz NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.chat_job_events (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  job_id uuid NOT NULL REFERENCES public.chat_jobs(id) ON DELETE CASCADE,
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.chat_browser_sessions ADD COLUMN IF NOT EXISTS viewport_width integer NOT NULL DEFAULT 1280 CHECK (viewport_width BETWEEN 1 AND 10000);
ALTER TABLE public.chat_browser_sessions ADD COLUMN IF NOT EXISTS viewport_height integer NOT NULL DEFAULT 1029 CHECK (viewport_height BETWEEN 1 AND 10000);
ALTER TABLE public.chat_browser_sessions ADD COLUMN IF NOT EXISTS agent_busy boolean NOT NULL DEFAULT false;

CREATE TABLE IF NOT EXISTS public.agent_operator_messages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id uuid NOT NULL REFERENCES public.chat_jobs(id) ON DELETE CASCADE,
  content text NOT NULL CHECK (length(content) BETWEEN 1 AND 8000),
  consumed boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.agent_tasks (
  id uuid PRIMARY KEY,
  objective text NOT NULL,
  outcome text NOT NULL,
  steps integer NOT NULL DEFAULT 0 CHECK (steps >= 0),
  verified boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.agent_reflections (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id uuid NOT NULL REFERENCES public.agent_tasks(id) ON DELETE CASCADE,
  objective text NOT NULL,
  failure_point text NOT NULL,
  component text NOT NULL CHECK (component IN ('planner', 'executor', 'tool', 'environment', 'verifier', 'model')),
  root_cause text NOT NULL,
  pivot_action text NOT NULL,
  patch text NOT NULL DEFAULT '',
  failed_action text NOT NULL DEFAULT '',
  corrective_action text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.agent_skills (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  dsl text NOT NULL,
  terms text NOT NULL DEFAULT '',
  vector double precision[] NOT NULL DEFAULT ARRAY[]::double precision[],
  successes integer NOT NULL DEFAULT 0 CHECK (successes >= 0),
  failures integer NOT NULL DEFAULT 0 CHECK (failures >= 0),
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'rejected')),
  baseline_rate double precision NULL,
  candidate_rate double precision NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.agent_playbooks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  content text NOT NULL,
  task_count integer NOT NULL CHECK (task_count >= 0),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.agent_policy_weights (
  ngram text PRIMARY KEY,
  weight double precision NOT NULL DEFAULT 0,
  observations integer NOT NULL DEFAULT 0 CHECK (observations >= 0),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chat_jobs_conversation_created_idx ON public.chat_jobs(conversation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS chat_jobs_running_lease_idx ON public.chat_jobs(status, lease_expires_at) WHERE status IN ('queued', 'running');
CREATE INDEX IF NOT EXISTS chat_job_events_job_id_idx ON public.chat_job_events(job_id, id);
CREATE INDEX IF NOT EXISTS chat_browser_sessions_job_created_idx ON public.chat_browser_sessions(job_id, created_at DESC);
CREATE INDEX IF NOT EXISTS agent_operator_messages_task_pending_idx ON public.agent_operator_messages(task_id, created_at) WHERE consumed = false;
CREATE INDEX IF NOT EXISTS agent_reflections_component_created_idx ON public.agent_reflections(component, created_at DESC);
CREATE INDEX IF NOT EXISTS agent_skills_status_idx ON public.agent_skills(status, created_at DESC);

ALTER TABLE public.chat_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_job_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_browser_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_operator_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_reflections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_skills ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_playbooks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_policy_weights ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS chat_conversations_owner_select ON public.chat_conversations;
CREATE POLICY chat_conversations_owner_select ON public.chat_conversations FOR SELECT TO authenticated USING (owner_id = auth.uid());
DROP POLICY IF EXISTS chat_jobs_owner_select ON public.chat_jobs;
CREATE POLICY chat_jobs_owner_select ON public.chat_jobs FOR SELECT TO authenticated USING (EXISTS (SELECT 1 FROM public.chat_conversations c WHERE c.id = conversation_id AND c.owner_id = auth.uid()));
DROP POLICY IF EXISTS chat_browser_sessions_owner_select ON public.chat_browser_sessions;
CREATE POLICY chat_browser_sessions_owner_select ON public.chat_browser_sessions FOR SELECT TO authenticated USING (EXISTS (SELECT 1 FROM public.chat_jobs j JOIN public.chat_conversations c ON c.id = j.conversation_id WHERE j.id = job_id AND c.owner_id = auth.uid()));

GRANT ALL ON public.chat_conversations, public.chat_jobs, public.chat_job_events, public.chat_browser_sessions, public.agent_operator_messages, public.agent_tasks, public.agent_reflections, public.agent_skills, public.agent_playbooks, public.agent_policy_weights TO service_role;
GRANT SELECT ON public.chat_conversations, public.chat_jobs, public.chat_browser_sessions TO authenticated;

CREATE OR REPLACE FUNCTION public.claim_chat_browser_step(_job_id uuid)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  PERFORM id FROM public.chat_browser_sessions WHERE job_id = _job_id AND status = 'active' FOR UPDATE;
  IF EXISTS (SELECT 1 FROM public.chat_browser_sessions WHERE job_id = _job_id AND status = 'active' AND control_mode <> 'agent') THEN
    RETURN false;
  END IF;
  UPDATE public.chat_browser_sessions SET agent_busy = true WHERE job_id = _job_id AND status = 'active';
  RETURN true;
END;
$$;

REVOKE ALL ON FUNCTION public.claim_chat_browser_step(uuid) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.claim_chat_browser_step(uuid) TO service_role;
