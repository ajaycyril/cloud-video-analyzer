import { z } from "zod";
import { edgeMetricsSchema, localDetectionSchema } from "./schema";

const scoreSchema = z.number().finite().min(0).max(100);
const normalizedSchema = z.number().finite().min(0).max(1);
const nonNegativeIntSchema = z.number().int().min(0);

export const providerSchema = z.enum(["gemini", "openai", "nvidia"]);
export const videoModeSchema = z.enum(["industrial_general", "person_zone", "ppe", "safety", "operations"]);
export const videoSourceSchema = z.enum(["camera", "file", "sample"]);

export const zoneSchema = z.object({
  id: z.string().min(1),
  label: z.string().min(1),
  x: normalizedSchema,
  y: normalizedSchema,
  w: z.number().finite().min(0.02).max(1),
  h: z.number().finite().min(0.02).max(1),
});

export const sampledFrameSchema = z.object({
  imageDataUrl: z.string().min(100),
  timestampMs: z.number().finite().min(0),
  edgeMetrics: edgeMetricsSchema,
  localDetections: z.array(localDetectionSchema).max(20),
});

export const videoAnalysisRequestSchema = z.object({
  provider: providerSchema,
  mode: videoModeSchema,
  source: videoSourceSchema,
  objective: z.string().min(12).max(800),
  zones: z.array(zoneSchema).max(4),
  frames: z.array(sampledFrameSchema).min(1).max(6),
  sampling: z.object({
    requestedFps: z.number().finite().min(0.1).max(5),
    maxFrames: z.number().int().min(1).max(6),
    jpegQuality: z.number().finite().min(0.3).max(0.9),
    payloadBytes: nonNegativeIntSchema,
  }),
});

export const videoAnalysisResponseSchema = z.object({
  provider: providerSchema,
  model: z.string().min(1),
  headline: z.string().min(1).max(120),
  commentary: z.string().min(1).max(260),
  confidence: scoreSchema,
  scene: z.object({
    summary: z.string().min(1),
    environment: z.string().min(1),
    activity: z.string().min(1),
  }),
  objects: z.array(
    z.object({
      label: z.string().min(1),
      count: z.number().int().min(1),
      locations: z.array(z.string().min(1)),
      evidence: z.string().min(1),
    }),
  ),
  events: z.array(
    z.object({
      timestamp: z.string().min(1),
      description: z.string().min(1),
      importance: z.enum(["low", "medium", "high"]),
    }),
  ),
  risks: z.array(
    z.object({
      label: z.string().min(1),
      severity: z.enum(["low", "medium", "high"]),
      rationale: z.string().min(1),
    }),
  ),
  alerts: z.array(
    z.object({
      label: z.string().min(1),
      triggered: z.boolean(),
      severity: z.enum(["low", "medium", "high"]),
      evidence: z.string().min(1),
      zoneId: z.string().nullable(),
    }),
  ),
  recommendations: z.array(
    z.object({
      priority: z.number().int().min(1).max(5),
      action: z.string().min(1),
      reason: z.string().min(1),
      owner: z.enum(["human", "automation", "robot"]),
    }),
  ),
  timeline: z.array(
    z.object({
      frame: z.number().int().min(1),
      observation: z.string().min(1),
    }),
  ),
  edgeAssessment: z.object({
    framesAnalyzed: z.number().int().min(1),
    cloudFramesSkipped: z.number().int().min(0),
    costControl: z.string().min(1),
  }),
  usage: z.object({
    inputTokens: nonNegativeIntSchema,
    outputTokens: nonNegativeIntSchema,
    totalTokens: nonNegativeIntSchema,
    estimatedCostUsd: z.number().finite().min(0),
    latencyMs: nonNegativeIntSchema,
  }),
});

export type ProviderId = z.infer<typeof providerSchema>;
export type VideoMode = z.infer<typeof videoModeSchema>;
export type VideoSource = z.infer<typeof videoSourceSchema>;
export type Zone = z.infer<typeof zoneSchema>;
export type SampledFrame = z.infer<typeof sampledFrameSchema>;
export type VideoAnalysisRequest = z.infer<typeof videoAnalysisRequestSchema>;
export type VideoAnalysisResponse = z.infer<typeof videoAnalysisResponseSchema>;

const stringSchema = { type: "string" } as const;
const scoreJsonSchema = { type: "number", minimum: 0, maximum: 100 } as const;

export const videoAnalysisJsonSchema = {
  type: "object",
  additionalProperties: false,
  properties: {
    headline: stringSchema,
    commentary: stringSchema,
    confidence: scoreJsonSchema,
    scene: {
      type: "object",
      additionalProperties: false,
      properties: {
        summary: stringSchema,
        environment: stringSchema,
        activity: stringSchema,
      },
      required: ["summary", "environment", "activity"],
    },
    objects: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          label: stringSchema,
          count: { type: "integer", minimum: 1 },
          locations: { type: "array", items: stringSchema },
          evidence: stringSchema,
        },
        required: ["label", "count", "locations", "evidence"],
      },
    },
    events: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          timestamp: stringSchema,
          description: stringSchema,
          importance: { type: "string", enum: ["low", "medium", "high"] },
        },
        required: ["timestamp", "description", "importance"],
      },
    },
    risks: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          label: stringSchema,
          severity: { type: "string", enum: ["low", "medium", "high"] },
          rationale: stringSchema,
        },
        required: ["label", "severity", "rationale"],
      },
    },
    alerts: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          label: stringSchema,
          triggered: { type: "boolean" },
          severity: { type: "string", enum: ["low", "medium", "high"] },
          evidence: stringSchema,
          zoneId: { type: ["string", "null"] },
        },
        required: ["label", "triggered", "severity", "evidence", "zoneId"],
      },
    },
    recommendations: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          priority: { type: "integer", minimum: 1, maximum: 5 },
          action: stringSchema,
          reason: stringSchema,
          owner: { type: "string", enum: ["human", "automation", "robot"] },
        },
        required: ["priority", "action", "reason", "owner"],
      },
    },
    timeline: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          frame: { type: "integer", minimum: 1 },
          observation: stringSchema,
        },
        required: ["frame", "observation"],
      },
    },
    edgeAssessment: {
      type: "object",
      additionalProperties: false,
      properties: {
        framesAnalyzed: { type: "integer", minimum: 1 },
        cloudFramesSkipped: { type: "integer", minimum: 0 },
        costControl: stringSchema,
      },
      required: ["framesAnalyzed", "cloudFramesSkipped", "costControl"],
    },
  },
  required: [
    "headline",
    "commentary",
    "confidence",
    "scene",
    "objects",
    "events",
    "risks",
    "alerts",
    "recommendations",
    "timeline",
    "edgeAssessment",
  ],
} as const;

export const geminiVideoResponseSchema = {
  type: "OBJECT",
  required: videoAnalysisJsonSchema.required,
  propertyOrdering: videoAnalysisJsonSchema.required,
  properties: {
    headline: { type: "STRING" },
    commentary: { type: "STRING" },
    confidence: { type: "NUMBER", minimum: 0, maximum: 100 },
    scene: {
      type: "OBJECT",
      required: ["summary", "environment", "activity"],
      propertyOrdering: ["summary", "environment", "activity"],
      properties: {
        summary: { type: "STRING" },
        environment: { type: "STRING" },
        activity: { type: "STRING" },
      },
    },
    objects: {
      type: "ARRAY",
      items: {
        type: "OBJECT",
        required: ["label", "count", "locations", "evidence"],
        propertyOrdering: ["label", "count", "locations", "evidence"],
        properties: {
          label: { type: "STRING" },
          count: { type: "INTEGER", minimum: 1 },
          locations: { type: "ARRAY", items: { type: "STRING" } },
          evidence: { type: "STRING" },
        },
      },
    },
    events: {
      type: "ARRAY",
      items: {
        type: "OBJECT",
        required: ["timestamp", "description", "importance"],
        propertyOrdering: ["timestamp", "description", "importance"],
        properties: {
          timestamp: { type: "STRING" },
          description: { type: "STRING" },
          importance: { type: "STRING", format: "enum", enum: ["low", "medium", "high"] },
        },
      },
    },
    risks: {
      type: "ARRAY",
      items: {
        type: "OBJECT",
        required: ["label", "severity", "rationale"],
        propertyOrdering: ["label", "severity", "rationale"],
        properties: {
          label: { type: "STRING" },
          severity: { type: "STRING", format: "enum", enum: ["low", "medium", "high"] },
          rationale: { type: "STRING" },
        },
      },
    },
    alerts: {
      type: "ARRAY",
      items: {
        type: "OBJECT",
        required: ["label", "triggered", "severity", "evidence", "zoneId"],
        propertyOrdering: ["label", "triggered", "severity", "evidence", "zoneId"],
        properties: {
          label: { type: "STRING" },
          triggered: { type: "BOOLEAN" },
          severity: { type: "STRING", format: "enum", enum: ["low", "medium", "high"] },
          evidence: { type: "STRING" },
          zoneId: { type: "STRING", nullable: true },
        },
      },
    },
    recommendations: {
      type: "ARRAY",
      items: {
        type: "OBJECT",
        required: ["priority", "action", "reason", "owner"],
        propertyOrdering: ["priority", "action", "reason", "owner"],
        properties: {
          priority: { type: "INTEGER", minimum: 1, maximum: 5 },
          action: { type: "STRING" },
          reason: { type: "STRING" },
          owner: { type: "STRING", format: "enum", enum: ["human", "automation", "robot"] },
        },
      },
    },
    timeline: {
      type: "ARRAY",
      items: {
        type: "OBJECT",
        required: ["frame", "observation"],
        propertyOrdering: ["frame", "observation"],
        properties: {
          frame: { type: "INTEGER", minimum: 1 },
          observation: { type: "STRING" },
        },
      },
    },
    edgeAssessment: {
      type: "OBJECT",
      required: ["framesAnalyzed", "cloudFramesSkipped", "costControl"],
      propertyOrdering: ["framesAnalyzed", "cloudFramesSkipped", "costControl"],
      properties: {
        framesAnalyzed: { type: "INTEGER", minimum: 1 },
        cloudFramesSkipped: { type: "INTEGER", minimum: 0 },
        costControl: { type: "STRING" },
      },
    },
  },
} as const;
