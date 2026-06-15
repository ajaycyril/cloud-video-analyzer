import { z } from "zod";

const scoreSchema = z.number().finite().min(0).max(100);
const normalizedSchema = z.number().finite().min(0).max(1);
const nonNegativeIntSchema = z.number().int().min(0);

export const modeSchema = z.enum(["space_scan", "desk_productivity", "webcam_coach"]);
export const scanPhaseSchema = z.enum(["observe", "baseline", "verify"]);

export const edgeMetricsSchema = z.object({
  brightness: scoreSchema,
  contrast: scoreSchema,
  sharpness: scoreSchema,
  edgeDensity: scoreSchema,
  visualComplexity: scoreSchema,
  frameDifference: scoreSchema,
  motionScore: scoreSchema,
  stability: scoreSchema,
  usable: z.boolean(),
  rejectionReason: z.string().nullable(),
});

export const localDetectionSchema = z.object({
  label: z.string().min(1),
  score: normalizedSchema,
  x: normalizedSchema,
  y: normalizedSchema,
  w: normalizedSchema,
  h: normalizedSchema,
});

export const worldStateSchema = z.object({
  summary: z.string().min(1),
  objects: z.array(
    z.object({
      id: z.string().min(1),
      label: z.string().min(1),
      location: z.string().min(1),
      affordance: z.string().min(1),
      movable: z.boolean(),
      source: z.enum(["local_detection", "gemini_reasoning"]),
    }),
  ),
  relationships: z.array(
    z.object({
      subject: z.string().min(1),
      relation: z.string().min(1),
      object: z.string().min(1),
      implication: z.string().min(1),
    }),
  ),
});

export const analysisResponseSchema = z.object({
  commentary: z.string().min(1).max(220),
  score: scoreSchema,
  scoreLabel: z.string().min(1),
  subscores: z.object({
    surfaceClarity: scoreSchema,
    objectGrouping: scoreSchema,
    spillRisk: scoreSchema,
    cableMess: scoreSchema,
    focusZone: scoreSchema,
    visualCalm: scoreSchema,
    lighting: scoreSchema,
    framing: scoreSchema,
    detectionConfidence: scoreSchema,
  }),
  worldState: worldStateSchema,
  events: z.array(z.string()),
  actions: z.array(
    z.object({
      priority: z.number().int().min(1).max(10),
      instruction: z.string().min(1),
      reason: z.string().min(1),
      expectedGain: z.number().finite().min(0).max(100),
      status: z.enum(["pending", "resolved", "unknown"]),
    }),
  ),
  overlays: z.array(
    z.object({
      type: z.enum(["zone", "label", "arrow", "detection"]),
      label: z.string().min(1),
      severity: z.enum(["low", "medium", "high", "positive"]),
      x: normalizedSchema,
      y: normalizedSchema,
      w: normalizedSchema.optional(),
      h: normalizedSchema.optional(),
      toX: normalizedSchema.optional(),
      toY: normalizedSchema.optional(),
    }),
  ),
  verification: z.object({
    baselineScore: scoreSchema.nullable(),
    currentScore: scoreSchema,
    predictedScore: scoreSchema.nullable(),
    delta: z.number().finite().min(-100).max(100).nullable(),
    resolved: z.array(z.string()),
    remaining: z.array(z.string()),
  }),
  usage: z
    .object({
      inputTokens: nonNegativeIntSchema,
      outputTokens: nonNegativeIntSchema,
      totalTokens: nonNegativeIntSchema,
      estimatedCostUsd: z.number().finite().min(0),
    })
    .optional(),
});

export const analyzeFrameRequestSchema = z.object({
  frameBase64: z.string().min(1),
  mode: modeSchema,
  scanPhase: scanPhaseSchema,
  previousWorldState: worldStateSchema.nullable(),
  baseline: analysisResponseSchema.nullable(),
  edgeMetrics: edgeMetricsSchema,
  localDetections: z.array(localDetectionSchema).max(20),
});

export const geminiResponseJsonSchema = {
  type: "object",
  properties: {
    commentary: { type: "string" },
    score: { type: "number" },
    scoreLabel: { type: "string" },
    subscores: {
      type: "object",
      properties: {
        surfaceClarity: { type: "number" },
        objectGrouping: { type: "number" },
        spillRisk: { type: "number" },
        cableMess: { type: "number" },
        focusZone: { type: "number" },
        visualCalm: { type: "number" },
        lighting: { type: "number" },
        framing: { type: "number" },
        detectionConfidence: { type: "number" },
      },
      required: ["surfaceClarity", "objectGrouping", "spillRisk", "cableMess", "focusZone", "visualCalm", "lighting", "framing", "detectionConfidence"],
    },
    worldState: {
      type: "object",
      properties: {
        summary: { type: "string" },
        objects: {
          type: "array",
          items: {
            type: "object",
            properties: {
              id: { type: "string" },
              label: { type: "string" },
              location: { type: "string" },
              affordance: { type: "string" },
              movable: { type: "boolean" },
              source: { type: "string", enum: ["local_detection", "gemini_reasoning"] },
            },
            required: ["id", "label", "location", "affordance", "movable", "source"],
          },
        },
        relationships: {
          type: "array",
          items: {
            type: "object",
            properties: {
              subject: { type: "string" },
              relation: { type: "string" },
              object: { type: "string" },
              implication: { type: "string" },
            },
            required: ["subject", "relation", "object", "implication"],
          },
        },
      },
      required: ["summary", "objects", "relationships"],
    },
    events: { type: "array", items: { type: "string" } },
    actions: {
      type: "array",
      items: {
        type: "object",
        properties: {
          priority: { type: "number" },
          instruction: { type: "string" },
          reason: { type: "string" },
          expectedGain: { type: "number" },
          status: { type: "string", enum: ["pending", "resolved", "unknown"] },
        },
        required: ["priority", "instruction", "reason", "expectedGain", "status"],
      },
    },
    overlays: {
      type: "array",
      items: {
        type: "object",
        properties: {
          type: { type: "string", enum: ["zone", "label", "arrow", "detection"] },
          label: { type: "string" },
          severity: { type: "string", enum: ["low", "medium", "high", "positive"] },
          x: { type: "number" },
          y: { type: "number" },
          w: { type: "number" },
          h: { type: "number" },
          toX: { type: "number" },
          toY: { type: "number" },
        },
        required: ["type", "label", "severity", "x", "y"],
      },
    },
    verification: {
      type: "object",
      properties: {
        baselineScore: { type: ["number", "null"] },
        currentScore: { type: "number" },
        predictedScore: { type: ["number", "null"] },
        delta: { type: ["number", "null"] },
        resolved: { type: "array", items: { type: "string" } },
        remaining: { type: "array", items: { type: "string" } },
      },
      required: ["baselineScore", "currentScore", "predictedScore", "delta", "resolved", "remaining"],
    },
  },
  required: ["commentary", "score", "scoreLabel", "subscores", "worldState", "events", "actions", "overlays", "verification"],
} as const;

const scoreResponseSchema = { type: "NUMBER", minimum: 0, maximum: 100 } as const;
const normalizedResponseSchema = { type: "NUMBER", minimum: 0, maximum: 1 } as const;
const stringArrayResponseSchema = { type: "ARRAY", items: { type: "STRING" } } as const;

export const geminiResponseSchema = {
  type: "OBJECT",
  required: ["commentary", "score", "scoreLabel", "subscores", "worldState", "events", "actions", "overlays", "verification"],
  propertyOrdering: ["commentary", "score", "scoreLabel", "subscores", "worldState", "events", "actions", "overlays", "verification"],
  properties: {
    commentary: {
      type: "STRING",
      description: "Concise live commentary under 28 words.",
    },
    score: scoreResponseSchema,
    scoreLabel: { type: "STRING" },
    subscores: {
      type: "OBJECT",
      required: ["surfaceClarity", "objectGrouping", "spillRisk", "cableMess", "focusZone", "visualCalm", "lighting", "framing", "detectionConfidence"],
      propertyOrdering: ["surfaceClarity", "objectGrouping", "spillRisk", "cableMess", "focusZone", "visualCalm", "lighting", "framing", "detectionConfidence"],
      properties: {
        surfaceClarity: scoreResponseSchema,
        objectGrouping: scoreResponseSchema,
        spillRisk: scoreResponseSchema,
        cableMess: scoreResponseSchema,
        focusZone: scoreResponseSchema,
        visualCalm: scoreResponseSchema,
        lighting: scoreResponseSchema,
        framing: scoreResponseSchema,
        detectionConfidence: scoreResponseSchema,
      },
    },
    worldState: {
      type: "OBJECT",
      required: ["summary", "objects", "relationships"],
      propertyOrdering: ["summary", "objects", "relationships"],
      properties: {
        summary: { type: "STRING" },
        objects: {
          type: "ARRAY",
          items: {
            type: "OBJECT",
            required: ["id", "label", "location", "affordance", "movable", "source"],
            propertyOrdering: ["id", "label", "location", "affordance", "movable", "source"],
            properties: {
              id: { type: "STRING" },
              label: { type: "STRING" },
              location: { type: "STRING" },
              affordance: { type: "STRING" },
              movable: { type: "BOOLEAN" },
              source: { type: "STRING", format: "enum", enum: ["local_detection", "gemini_reasoning"] },
            },
          },
        },
        relationships: {
          type: "ARRAY",
          items: {
            type: "OBJECT",
            required: ["subject", "relation", "object", "implication"],
            propertyOrdering: ["subject", "relation", "object", "implication"],
            properties: {
              subject: { type: "STRING" },
              relation: { type: "STRING" },
              object: { type: "STRING" },
              implication: { type: "STRING" },
            },
          },
        },
      },
    },
    events: stringArrayResponseSchema,
    actions: {
      type: "ARRAY",
      items: {
        type: "OBJECT",
        required: ["priority", "instruction", "reason", "expectedGain", "status"],
        propertyOrdering: ["priority", "instruction", "reason", "expectedGain", "status"],
        properties: {
          priority: { type: "INTEGER", minimum: 1, maximum: 10 },
          instruction: { type: "STRING" },
          reason: { type: "STRING" },
          expectedGain: scoreResponseSchema,
          status: { type: "STRING", format: "enum", enum: ["pending", "resolved", "unknown"] },
        },
      },
    },
    overlays: {
      type: "ARRAY",
      items: {
        type: "OBJECT",
        required: ["type", "label", "severity", "x", "y"],
        propertyOrdering: ["type", "label", "severity", "x", "y", "w", "h", "toX", "toY"],
        properties: {
          type: { type: "STRING", format: "enum", enum: ["zone", "label", "arrow", "detection"] },
          label: { type: "STRING" },
          severity: { type: "STRING", format: "enum", enum: ["low", "medium", "high", "positive"] },
          x: normalizedResponseSchema,
          y: normalizedResponseSchema,
          w: { ...normalizedResponseSchema, nullable: true },
          h: { ...normalizedResponseSchema, nullable: true },
          toX: { ...normalizedResponseSchema, nullable: true },
          toY: { ...normalizedResponseSchema, nullable: true },
        },
      },
    },
    verification: {
      type: "OBJECT",
      required: ["baselineScore", "currentScore", "predictedScore", "delta", "resolved", "remaining"],
      propertyOrdering: ["baselineScore", "currentScore", "predictedScore", "delta", "resolved", "remaining"],
      properties: {
        baselineScore: { ...scoreResponseSchema, nullable: true },
        currentScore: scoreResponseSchema,
        predictedScore: { ...scoreResponseSchema, nullable: true },
        delta: { type: "NUMBER", minimum: -100, maximum: 100, nullable: true },
        resolved: stringArrayResponseSchema,
        remaining: stringArrayResponseSchema,
      },
    },
  },
} as const;
