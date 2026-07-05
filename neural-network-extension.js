(function (Scratch) {
  "use strict";

  if (!Scratch) {
    throw new Error("This extension must be run inside TurboWarp or Scratch.");
  }

  const DEFAULT_NAME = "net";
  const networks = Object.create(null);
  const STAGE_HALF_WIDTH = 240;
  const STAGE_HALF_HEIGHT = 180;
  const viewer = {
    installed: false,
    visible: false,
    selectedName: DEFAULT_NAME,
    button: null,
    panel: null,
    timer: null,
  };
  const trainingGhosts = {
    layer: null,
    timer: null,
  };
  const pretrainView = {
    layer: null,
  };

  const clampInteger = (value, min, max) => {
    const number = Math.floor(Number(value));
    if (!Number.isFinite(number)) return min;
    return Math.max(min, Math.min(max, number));
  };

  const clampNumber = (value, fallback, min, max) => {
    const number = Number(value);
    if (!Number.isFinite(number)) return fallback;
    return Math.max(min, Math.min(max, number));
  };

  const cleanName = (name) => {
    const text = String(name || "").trim();
    return text || DEFAULT_NAME;
  };

  const parseNumberList = (value) => {
    if (Array.isArray(value)) return value.map(Number).filter(Number.isFinite);

    const text = String(value || "").trim();
    if (!text) return [];

    try {
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed)) {
        return parsed.map(Number).filter(Number.isFinite);
      }
    } catch (_error) {
      // Comma/space parsing below is friendlier for Scratch blocks.
    }

    return text
      .split(/[\s,;]+/)
      .map(Number)
      .filter(Number.isFinite);
  };

  const formatNumberList = (values) =>
    values.map((value) => Number(value.toFixed(6))).join(", ");

  const activationValue = (kind, x) => {
    switch (kind) {
      case "sigmoid":
        return 1 / (1 + Math.exp(-x));
      case "relu":
        return Math.max(0, x);
      case "linear":
        return x;
      case "tanh":
      default:
        return Math.tanh(x);
    }
  };

  const activationDerivative = (kind, activatedValue) => {
    switch (kind) {
      case "sigmoid":
        return activatedValue * (1 - activatedValue);
      case "relu":
        return activatedValue > 0 ? 1 : 0;
      case "linear":
        return 1;
      case "tanh":
      default:
        return 1 - activatedValue * activatedValue;
    }
  };

  const randomWeight = (fanIn, fanOut) => {
    const limit = Math.sqrt(6 / (fanIn + fanOut));
    return (Math.random() * 2 - 1) * limit;
  };

  const makeLayer = (inputSize, outputSize) => ({
    weights: Array.from({ length: outputSize }, () =>
      Array.from({ length: inputSize }, () => randomWeight(inputSize, outputSize))
    ),
    biases: Array.from({ length: outputSize }, () => 0),
  });

  const makeNetwork = (inputSize, hiddenSizes, outputSize, learningRate, activation) => {
    const sizes = [inputSize, ...hiddenSizes, outputSize];
    return {
      inputSize,
      hiddenSizes,
      outputSize,
      learningRate,
      activation,
      layers: sizes.slice(1).map((size, index) => makeLayer(sizes[index], size)),
      samples: [],
      lastLoss: 0,
      goal: { type: "point", x: 0, y: 0, spriteName: "" },
      fail: {
        spriteName: "",
        timeLimit: 0,
        startedAt: Date.now(),
        failed: false,
        reason: "",
        recoveryUntilClear: false,
      },
      trainingGhosts: {
        visible: false,
        limit: 150,
        ghost: 70,
      },
      pretrain: {
        success: false,
        rounds: 0,
        reason: "not run",
        steps: 0,
        visible: false,
        paths: [],
      },
      recovery: {
        count: 0,
        reason: "not run",
        samples: 0,
        epochs: 0,
      },
      customMove: {
        requested: false,
        reason: "not requested",
        desiredDirection: 90,
        desiredTurn: 0,
        desiredX: 0,
        desiredY: 0,
        speed: 4,
        train: 0,
      },
      lastInputs: [],
      lastPrediction: [],
      lastActivations: [],
    };
  };

  const normalizeInputs = (network, values) => {
    const result = values.slice(0, network.inputSize);
    while (result.length < network.inputSize) result.push(0);
    return result;
  };

  const normalizeOutputs = (network, values) => {
    const result = values.slice(0, network.outputSize);
    while (result.length < network.outputSize) result.push(0);
    return result;
  };

  const forward = (network, inputs) => {
    const activations = [normalizeInputs(network, inputs)];

    for (let layerIndex = 0; layerIndex < network.layers.length; layerIndex += 1) {
      const layer = network.layers[layerIndex];
      const previous = activations[layerIndex];
      const isOutputLayer = layerIndex === network.layers.length - 1;
      const kind = isOutputLayer ? "linear" : network.activation;

      const current = layer.weights.map((weights, neuronIndex) => {
        let sum = layer.biases[neuronIndex];
        for (let i = 0; i < weights.length; i += 1) {
          sum += weights[i] * previous[i];
        }
        return activationValue(kind, sum);
      });

      activations.push(current);
    }

    network.lastInputs = activations[0].slice();
    network.lastPrediction = activations[activations.length - 1].slice();
    network.lastActivations = activations.map((values) => values.slice());
    return activations;
  };

  const trainSample = (network, inputs, expectedOutputs) => {
    const targets = normalizeOutputs(network, expectedOutputs);
    const activations = forward(network, inputs);
    const deltas = new Array(network.layers.length);
    const prediction = activations[activations.length - 1];
    let loss = 0;

    deltas[deltas.length - 1] = prediction.map((value, index) => {
      const error = value - targets[index];
      loss += error * error;
      return error;
    });

    for (let layerIndex = network.layers.length - 2; layerIndex >= 0; layerIndex -= 1) {
      const nextLayer = network.layers[layerIndex + 1];
      const nextDelta = deltas[layerIndex + 1];
      const currentActivation = activations[layerIndex + 1];

      deltas[layerIndex] = currentActivation.map((activatedValue, neuronIndex) => {
        let error = 0;
        for (let nextNeuron = 0; nextNeuron < nextDelta.length; nextNeuron += 1) {
          error += nextLayer.weights[nextNeuron][neuronIndex] * nextDelta[nextNeuron];
        }
        return error * activationDerivative(network.activation, activatedValue);
      });
    }

    for (let layerIndex = 0; layerIndex < network.layers.length; layerIndex += 1) {
      const layer = network.layers[layerIndex];
      const previous = activations[layerIndex];
      const delta = deltas[layerIndex];

      for (let neuronIndex = 0; neuronIndex < layer.weights.length; neuronIndex += 1) {
        for (let weightIndex = 0; weightIndex < layer.weights[neuronIndex].length; weightIndex += 1) {
          layer.weights[neuronIndex][weightIndex] -=
            network.learningRate * delta[neuronIndex] * previous[weightIndex];
        }
        layer.biases[neuronIndex] -= network.learningRate * delta[neuronIndex];
      }
    }

    return loss / network.outputSize;
  };

  const trainEpoch = (network) => {
    if (network.samples.length === 0) return 0;

    let loss = 0;
    for (const sample of network.samples) {
      loss += trainSample(network, sample.inputs, sample.outputs);
    }

    network.lastLoss = loss / network.samples.length;
    return network.lastLoss;
  };

  const getNetwork = (name) => networks[cleanName(name)];

  const ensureNetwork = (name) => {
    const networkName = cleanName(name);
    if (!networks[networkName]) {
      networks[networkName] = makeNetwork(1, [8], 1, 0.03, "tanh");
    }
    return networks[networkName];
  };

  const parseHiddenLayers = (value) => {
    const sizes = parseNumberList(value)
      .map((size) => clampInteger(size, 1, 128))
      .filter((size) => size > 0);
    return sizes.length ? sizes : [8];
  };

  const evaluateFormula = (formula, x) => {
    const expression = String(formula || "x").trim().replace(/\^/g, "**");
    if (!/^[0-9a-zA-Z_+\-*/%().,\s*]+$/.test(expression)) return 0;

    try {
      const evaluator = Function(
        "x",
        "sin",
        "cos",
        "tan",
        "abs",
        "sqrt",
        "round",
        "floor",
        "ceil",
        "min",
        "max",
        "pow",
        "log",
        "exp",
        "PI",
        "E",
        "pi",
        "e",
        `"use strict"; return (${expression});`
      );
      const value = evaluator(
        x,
        Math.sin,
        Math.cos,
        Math.tan,
        Math.abs,
        Math.sqrt,
        Math.round,
        Math.floor,
        Math.ceil,
        Math.min,
        Math.max,
        Math.pow,
        Math.log,
        Math.exp,
        Math.PI,
        Math.E,
        Math.PI,
        Math.E
      );
      return Number.isFinite(Number(value)) ? Number(value) : 0;
    } catch (_error) {
      return 0;
    }
  };

  const addFunctionSamples = (network, formula, start, end, count, owner) => {
    const sampleCount = clampInteger(count, 2, 10000);
    const first = Number(start);
    const last = Number(end);
    const from = Number.isFinite(first) ? first : -1;
    const to = Number.isFinite(last) ? last : 1;

    for (let i = 0; i < sampleCount; i += 1) {
      const x = sampleCount === 1 ? from : from + ((to - from) * i) / (sampleCount - 1);
      network.samples.push(ownedSample(network, [x], [evaluateFormula(formula, x)], owner));
    }
  };

  const getRuntime = (util) => {
    if (util && util.runtime) return util.runtime;
    if (Scratch.vm && Scratch.vm.runtime) return Scratch.vm.runtime;
    return null;
  };

  const targetName = (target) => {
    if (!target) return "";
    if (target.sprite && target.sprite.name) return target.sprite.name;
    if (typeof target.getName === "function") return target.getName();
    return target.name || "";
  };

  const ownerInfoFromUtil = (util) => {
    const target = util && util.target;
    if (!target) {
      return { key: "stage", id: "stage", name: "Stage", isClone: false };
    }

    const name = targetName(target) || "Sprite";
    const id =
      target.id ||
      target.targetId ||
      target.drawableID ||
      (target.isOriginal === false ? `clone-${name}` : `original-${name}`);
    return {
      key: `${name}:${id}`,
      id: String(id),
      name,
      isClone: target.isOriginal === false,
    };
  };

  const ownerQuery = (value, util) => {
    const text = String(value || "").trim();
    if (!text || text.toLowerCase() === "this" || text.toLowerCase() === "this sprite") {
      return ownerInfoFromUtil(util).key;
    }
    return text;
  };

  const sampleMatchesOwner = (sample, owner) => {
    if (!owner || owner.toLowerCase() === "all") return true;
    const meta = sample && sample.owner;
    if (!meta) return owner.toLowerCase() === "unowned";
    return meta.key === owner || meta.id === owner || meta.name === owner;
  };

  const ownedSample = (network, inputs, outputs, owner) => ({
    inputs: normalizeInputs(network, inputs),
    outputs: normalizeOutputs(network, outputs),
    owner,
    createdAt: Date.now(),
  });

  const findTargetByName = (name, util) => {
    const spriteName = String(name || "").trim();
    const runtime = getRuntime(util);
    if (!runtime || !Array.isArray(runtime.targets)) return null;
    return runtime.targets.find((target) => !target.isStage && targetName(target) === spriteName) || null;
  };

  const getSpriteMenu = (util) => {
    const runtime = getRuntime(util);
    if (!runtime || !Array.isArray(runtime.targets)) {
      return [{ text: "Sprite1", value: "Sprite1" }];
    }

    const items = runtime.targets
      .filter((target) => !target.isStage && target.isOriginal !== false)
      .map((target) => targetName(target))
      .filter(Boolean);

    return items.length ? items.map((name) => ({ text: name, value: name })) : ["Sprite1"];
  };

  const setTargetXY = (target, x, y) => {
    const nextX = clampNumber(x, 0, -STAGE_HALF_WIDTH, STAGE_HALF_WIDTH);
    const nextY = clampNumber(y, 0, -STAGE_HALF_HEIGHT, STAGE_HALF_HEIGHT);
    if (target && typeof target.setXY === "function") {
      target.setXY(nextX, nextY);
      return;
    }
    if (target) {
      target.x = nextX;
      target.y = nextY;
    }
  };

  const normalizedSpriteInputs = (spriteX, spriteY, goalX, goalY) => [
    clampNumber(spriteX / STAGE_HALF_WIDTH, 0, -1, 1),
    clampNumber(spriteY / STAGE_HALF_HEIGHT, 0, -1, 1),
    clampNumber(goalX / STAGE_HALF_WIDTH, 0, -1, 1),
    clampNumber(goalY / STAGE_HALF_HEIGHT, 0, -1, 1),
  ];

  const directionToGoal = (spriteX, spriteY, goalX, goalY) => {
    const dx = goalX - spriteX;
    const dy = goalY - spriteY;
    const distance = Math.sqrt(dx * dx + dy * dy) || 1;
    return [dx / distance, dy / distance];
  };

  const directionToGoalAvoidingDanger = (spriteX, spriteY, goalX, goalY, network, util) => {
    const goalDirection = directionToGoal(spriteX, spriteY, goalX, goalY);
    const fail = network.fail || {};
    const danger = fail.spriteName ? findTargetByName(fail.spriteName, util) : null;

    if (!danger) return goalDirection;

    const dangerDx = spriteX - (danger.x || 0);
    const dangerDy = spriteY - (danger.y || 0);
    const dangerDistance = Math.sqrt(dangerDx * dangerDx + dangerDy * dangerDy) || 1;
    const dangerRadius = 95;

    if (dangerDistance >= dangerRadius) return goalDirection;

    const force = (dangerRadius - dangerDistance) / dangerRadius;
    const mixedX = goalDirection[0] + (dangerDx / dangerDistance) * force * 2.2;
    const mixedY = goalDirection[1] + (dangerDy / dangerDistance) * force * 2.2;
    const mixedDistance = Math.sqrt(mixedX * mixedX + mixedY * mixedY) || 1;
    return [mixedX / mixedDistance, mixedY / mixedDistance];
  };

  const ensureControllerNetwork = (name, hiddenSizes, rate) => {
    const networkName = cleanName(name);
    const existing = networks[networkName];
    if (!existing || existing.inputSize !== 4 || existing.outputSize !== 2) {
      networks[networkName] = makeNetwork(
        4,
        hiddenSizes && hiddenSizes.length ? hiddenSizes : [12, 8],
        2,
        clampNumber(rate, 0.04, 0.000001, 1),
        "tanh"
      );
    } else if (Number.isFinite(Number(rate))) {
      existing.learningRate = clampNumber(rate, existing.learningRate, 0.000001, 1);
    }
    return networks[networkName];
  };

  const goalPosition = (network, util) => {
    const goal = network.goal || { type: "point", x: 0, y: 0, spriteName: "" };
    if (goal.type === "sprite") {
      const target = findTargetByName(goal.spriteName, util);
      if (target) return { x: target.x || 0, y: target.y || 0, spriteName: goal.spriteName };
    }
    return { x: Number(goal.x) || 0, y: Number(goal.y) || 0, spriteName: goal.spriteName || "" };
  };

  const addControllerSamples = (network, count, util) => {
    const samples = clampInteger(count, 1, 10000);
    const goal = goalPosition(network, util);
    const owner = ownerInfoFromUtil(util);

    for (let i = 0; i < samples; i += 1) {
      const spriteX = Math.random() * STAGE_HALF_WIDTH * 2 - STAGE_HALF_WIDTH;
      const spriteY = Math.random() * STAGE_HALF_HEIGHT * 2 - STAGE_HALF_HEIGHT;
      network.samples.push(
        ownedSample(
          network,
          normalizedSpriteInputs(spriteX, spriteY, goal.x, goal.y),
          directionToGoalAvoidingDanger(spriteX, spriteY, goal.x, goal.y, network, util),
          owner
        )
      );
    }
  };

  const addFocusedControllerSamples = (network, count, startState, util) => {
    const samples = clampInteger(count, 0, 10000);
    if (samples <= 0) return;

    const goal = goalPosition(network, util);
    const fail = network.fail || {};
    const danger = fail.spriteName ? findTargetByName(fail.spriteName, util) : null;
    const owner = ownerInfoFromUtil(util);

    for (let i = 0; i < samples; i += 1) {
      let spriteX;
      let spriteY;

      if (danger && i % 2 === 0) {
        const angle = Math.random() * Math.PI * 2;
        const radius = 25 + Math.random() * 95;
        spriteX = (danger.x || 0) + Math.cos(angle) * radius;
        spriteY = (danger.y || 0) + Math.sin(angle) * radius;
      } else {
        const amount = Math.random();
        const wiggleX = Math.random() * 100 - 50;
        const wiggleY = Math.random() * 100 - 50;
        spriteX = startState.x + (goal.x - startState.x) * amount + wiggleX;
        spriteY = startState.y + (goal.y - startState.y) * amount + wiggleY;
      }

      spriteX = clampNumber(spriteX, 0, -STAGE_HALF_WIDTH, STAGE_HALF_WIDTH);
      spriteY = clampNumber(spriteY, 0, -STAGE_HALF_HEIGHT, STAGE_HALF_HEIGHT);
      network.samples.push(
        ownedSample(
          network,
          normalizedSpriteInputs(spriteX, spriteY, goal.x, goal.y),
          directionToGoalAvoidingDanger(spriteX, spriteY, goal.x, goal.y, network, util),
          owner
        )
      );
    }
  };

  const scratchDirectionToVector = (direction) => {
    const radians = ((Number(direction) || 90) * Math.PI) / 180;
    return [Math.sin(radians), Math.cos(radians)];
  };

  const vectorToScratchDirection = (x, y) => {
    const direction = (Math.atan2(x, y) * 180) / Math.PI;
    return Number.isFinite(direction) ? direction : 90;
  };

  const angleDifference = (from, to) => {
    let difference = ((to - from + 180) % 360) - 180;
    if (difference < -180) difference += 360;
    return difference;
  };

  const setTargetDirection = (target, direction) => {
    if (target && typeof target.setDirection === "function") {
      target.setDirection(direction);
      return;
    }
    if (target) target.direction = direction;
  };

  const moveTargetForward = (target, speed) => {
    const direction = Number.isFinite(Number(target.direction)) ? Number(target.direction) : 90;
    const vector = scratchDirectionToVector(direction);
    setTargetXY(target, (target.x || 0) + vector[0] * speed, (target.y || 0) + vector[1] * speed);
  };

  const dangerHitAt = (network, x, y, util) => {
    const fail = network.fail || {};
    if (!fail.spriteName) return false;
    const danger = findTargetByName(fail.spriteName, util);
    if (!danger) return false;
    const dx = x - (danger.x || 0);
    const dy = y - (danger.y || 0);
    return Math.sqrt(dx * dx + dy * dy) < 24;
  };

  const goalReachedAt = (network, x, y, util) => {
    const goal = goalPosition(network, util);
    const dx = x - goal.x;
    const dy = y - goal.y;
    return Math.sqrt(dx * dx + dy * dy) < (network.goal && network.goal.type === "sprite" ? 24 : 12);
  };

  const simulateSteeringRun = (network, startState, util, options) => {
    const speed = clampNumber(options.speed, 4, 0, 50);
    const maxTurn = clampNumber(options.turn, 12, 0, 180);
    const steps = clampInteger(options.steps, 1, 2000);
    let x = clampNumber(startState.x, 0, -STAGE_HALF_WIDTH, STAGE_HALF_WIDTH);
    let y = clampNumber(startState.y, 0, -STAGE_HALF_HEIGHT, STAGE_HALF_HEIGHT);
    let direction = Number.isFinite(Number(startState.direction)) ? Number(startState.direction) : 90;
    const goal = goalPosition(network, util);
    const path = [{ x, y }];

    for (let step = 0; step <= steps; step += 1) {
      if (goalReachedAt(network, x, y, util)) {
        return { success: true, reason: "reached goal", steps: step, x, y, direction, path };
      }
      if (dangerHitAt(network, x, y, util)) {
        return { success: false, reason: "touching danger", steps: step, x, y, direction, path };
      }

      const activations = forward(network, normalizedSpriteInputs(x, y, goal.x, goal.y));
      const output = activations[activations.length - 1];
      const desiredDirection = vectorToScratchDirection(
        clampNumber(output[0], 0, -1, 1),
        clampNumber(output[1], 0, -1, 1)
      );
      const turn = clampNumber(angleDifference(direction, desiredDirection), 0, -maxTurn, maxTurn);
      direction += turn;
      const vector = scratchDirectionToVector(direction);
      x = clampNumber(x + vector[0] * speed, 0, -STAGE_HALF_WIDTH, STAGE_HALF_WIDTH);
      y = clampNumber(y + vector[1] * speed, 0, -STAGE_HALF_HEIGHT, STAGE_HALF_HEIGHT);
      if (step % 2 === 0) path.push({ x, y });
    }

    return { success: false, reason: "too many steps", steps, x, y, direction, path };
  };

  const pretrainControllerFromSprite = (network, args, util) => {
    const target = util && util.target;
    const startState = {
      x: target && !target.isStage ? target.x || 0 : -180,
      y: target && !target.isStage ? target.y || 0 : -120,
      direction: target && Number.isFinite(Number(target.direction)) ? Number(target.direction) : 90,
    };
    const rounds = clampInteger(args.ROUNDS, 1, 200);
    const samples = clampInteger(args.SAMPLES, 1, 5000);
    const epochs = clampInteger(args.EPOCHS, 1, 10000);
    const speed = clampNumber(args.SPEED, 4, 0, 50);
    const turn = clampNumber(args.TURN, 12, 0, 180);
    const steps = Number.isFinite(Number(args.STEPS)) ? clampInteger(args.STEPS, 1, 2000) : 180;
    let result = { success: false, reason: "not run", steps: 0 };
    const paths = [];

    for (let round = 1; round <= rounds; round += 1) {
      addControllerSamples(network, Math.ceil(samples / 2), util);
      addFocusedControllerSamples(network, Math.floor(samples / 2), startState, util);
      for (let epoch = 0; epoch < epochs; epoch += 1) {
        trainEpoch(network);
      }

      result = simulateSteeringRun(network, startState, util, { speed, turn, steps });
      if (Array.isArray(result.path)) {
        paths.push({
          success: result.success,
          reason: result.reason,
          points: result.path,
        });
        while (paths.length > 20) paths.shift();
      }
      network.pretrain = {
        ...(network.pretrain || {}),
        success: result.success,
        rounds: round,
        reason: result.reason,
        steps: result.steps,
        paths,
      };
      if (result.success) break;
    }

    renderPretrainView();
    return network.pretrain;
  };

  const addFailureRecoverySamples = (network, count, failState, util) => {
    const samples = clampInteger(count, 0, 10000);
    if (samples <= 0) return;

    const goal = goalPosition(network, util);
    const fail = network.fail || {};
    const danger = fail.spriteName ? findTargetByName(fail.spriteName, util) : null;
    const owner = ownerInfoFromUtil(util);

    for (let i = 0; i < samples; i += 1) {
      let spriteX = failState.x;
      let spriteY = failState.y;

      if (danger && i % 3 !== 2) {
        const angle = Math.random() * Math.PI * 2;
        const radius = 18 + Math.random() * 90;
        spriteX = (danger.x || 0) + Math.cos(angle) * radius;
        spriteY = (danger.y || 0) + Math.sin(angle) * radius;
      } else {
        spriteX += Math.random() * 100 - 50;
        spriteY += Math.random() * 100 - 50;
      }

      spriteX = clampNumber(spriteX, 0, -STAGE_HALF_WIDTH, STAGE_HALF_WIDTH);
      spriteY = clampNumber(spriteY, 0, -STAGE_HALF_HEIGHT, STAGE_HALF_HEIGHT);
      network.samples.push(
        ownedSample(
          network,
          normalizedSpriteInputs(spriteX, spriteY, goal.x, goal.y),
          directionToGoalAvoidingDanger(spriteX, spriteY, goal.x, goal.y, network, util),
          owner
        )
      );
    }
  };

  const learnFromFailure = (network, args, util) => {
    const target = util && util.target;
    const reasonBeforeCheck = (network.fail && network.fail.reason) || "";
    checkFailConditions(network, util);
    const reason = (network.fail && network.fail.reason) || reasonBeforeCheck || "not failed";
    const samples = clampInteger(args.SAMPLES, 1, 10000);
    const epochs = clampInteger(args.EPOCHS, 1, 10000);
    const failState = {
      x: target && !target.isStage ? target.x || 0 : -180,
      y: target && !target.isStage ? target.y || 0 : -120,
      direction: target && Number.isFinite(Number(target.direction)) ? Number(target.direction) : 90,
    };

    addFailureRecoverySamples(network, Math.ceil(samples * 0.7), failState, util);
    addFocusedControllerSamples(network, Math.floor(samples * 0.3), failState, util);
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      trainEpoch(network);
    }

    network.recovery = {
      count: ((network.recovery && network.recovery.count) || 0) + 1,
      reason,
      samples,
      epochs,
    };

    network.fail = {
      ...(network.fail || {}),
      startedAt: Date.now(),
      failed: false,
      reason: "",
      recoveryUntilClear: reason.indexOf("touching") !== -1,
    };

    return network.recovery;
  };

  const exportTrainingPacket = (network, name, owner) => {
    const samples = network.samples
      .filter((sample) => sampleMatchesOwner(sample, owner))
      .map((sample) => ({
        inputs: normalizeInputs(network, Array.isArray(sample.inputs) ? sample.inputs : []),
        outputs: normalizeOutputs(network, Array.isArray(sample.outputs) ? sample.outputs : []),
        owner: sample.owner || null,
        createdAt: sample.createdAt || null,
      }));

    return JSON.stringify({
      type: "scratch-neurals-training",
      version: 1,
      network: cleanName(name),
      owner,
      inputSize: network.inputSize,
      outputSize: network.outputSize,
      count: samples.length,
      samples,
    });
  };

  const importTrainingPacket = (network, json, util) => {
    try {
      const packet = JSON.parse(String(json || "{}"));
      const incoming = Array.isArray(packet)
        ? packet
        : Array.isArray(packet.samples)
          ? packet.samples
          : [];
      const importer = ownerInfoFromUtil(util);
      let count = 0;

      for (const sample of incoming) {
        if (!sample || !Array.isArray(sample.inputs) || !Array.isArray(sample.outputs)) continue;
        network.samples.push({
          inputs: normalizeInputs(network, sample.inputs.map(Number).filter(Number.isFinite)),
          outputs: normalizeOutputs(network, sample.outputs.map(Number).filter(Number.isFinite)),
          owner: sample.owner || {
            key: `imported-by-${importer.key}`,
            id: importer.id,
            name: importer.name,
            isClone: importer.isClone,
          },
          importedBy: importer,
          importedAt: Date.now(),
          createdAt: sample.createdAt || Date.now(),
        });
        count += 1;
      }

      network.lastImport = {
        count,
        from: packet.owner || "unknown",
        importedBy: importer.key,
      };
      return count;
    } catch (_error) {
      network.lastImport = {
        count: 0,
        from: "invalid JSON",
        importedBy: ownerInfoFromUtil(util).key,
      };
      return 0;
    }
  };

  const runSteeringController = (args, util, defaultTurn) => {
    const network = ensureControllerNetwork(args.NAME);
    const target = util && util.target;
    if (!target || target.isStage) {
      network.customMove = {
        ...(network.customMove || {}),
        requested: false,
        reason: "no sprite",
      };
      return;
    }
    if (checkFailConditions(network, util)) {
      network.customMove = {
        ...(network.customMove || {}),
        requested: false,
        reason: (network.fail && network.fail.reason) || "failed",
      };
      return;
    }

    const trainSamples = clampInteger(args.TRAIN, 0, 1000);
    if (trainSamples > 0) {
      addControllerSamples(network, trainSamples, util);
      trainEpoch(network);
    }

    if (isGoalReached(network, util)) {
      network.customMove = {
        ...(network.customMove || {}),
        requested: false,
        reason: "goal reached",
      };
      return;
    }

    const goal = goalPosition(network, util);
    const inputs = normalizedSpriteInputs(target.x || 0, target.y || 0, goal.x, goal.y);
    const activations = forward(network, inputs);
    const output = activations[activations.length - 1];
    const desiredDirection = vectorToScratchDirection(
      clampNumber(output[0], 0, -1, 1),
      clampNumber(output[1], 0, -1, 1)
    );
    const currentDirection = Number.isFinite(Number(target.direction)) ? Number(target.direction) : 90;
    const maxTurn = clampNumber(args.TURN, defaultTurn, 0, 180);
    const turn = clampNumber(angleDifference(currentDirection, desiredDirection), 0, -maxTurn, maxTurn);
    const speed = clampNumber(args.SPEED, 4, 0, 50);
    network.customMove = {
      requested: true,
      reason: "move",
      desiredDirection,
      desiredTurn: turn,
      desiredX: clampNumber(output[0], 0, -1, 1),
      desiredY: clampNumber(output[1], 0, -1, 1),
      speed,
      train: trainSamples,
      turnLimit: maxTurn,
      owner: ownerInfoFromUtil(util),
      createdAt: Date.now(),
    };
    setTargetDirection(target, currentDirection + turn);
    moveTargetForward(target, speed);
    checkFailConditions(network, util);
  };

  const computeCustomMoveIntent = (args, util, defaultTurn) => {
    const network = ensureControllerNetwork(args.NAME);
    const target = util && util.target;
    if (!target || target.isStage) {
      network.customMove = {
        ...(network.customMove || {}),
        requested: false,
        reason: "no sprite",
      };
      return network.customMove;
    }
    if (checkFailConditions(network, util)) {
      network.customMove = {
        ...(network.customMove || {}),
        requested: false,
        reason: (network.fail && network.fail.reason) || "failed",
      };
      return network.customMove;
    }
    if (isGoalReached(network, util)) {
      network.customMove = {
        ...(network.customMove || {}),
        requested: false,
        reason: "goal reached",
      };
      return network.customMove;
    }

    const trainSamples = clampInteger(args.TRAIN, 0, 1000);
    if (trainSamples > 0) {
      addControllerSamples(network, trainSamples, util);
      trainEpoch(network);
    }

    const goal = goalPosition(network, util);
    const inputs = normalizedSpriteInputs(target.x || 0, target.y || 0, goal.x, goal.y);
    const activations = forward(network, inputs);
    const output = activations[activations.length - 1];
    const desiredX = clampNumber(output[0], 0, -1, 1);
    const desiredY = clampNumber(output[1], 0, -1, 1);
    const desiredDirection = vectorToScratchDirection(desiredX, desiredY);
    const currentDirection = Number.isFinite(Number(target.direction)) ? Number(target.direction) : 90;
    const maxTurn = clampNumber(args.TURN, defaultTurn, 0, 180);
    const desiredTurn = clampNumber(
      angleDifference(currentDirection, desiredDirection),
      0,
      -maxTurn,
      maxTurn
    );

    network.customMove = {
      requested: true,
      reason: "move",
      desiredDirection,
      desiredTurn,
      desiredX,
      desiredY,
      speed: clampNumber(args.SPEED, 4, 0, 50),
      train: trainSamples,
      turnLimit: maxTurn,
      owner: ownerInfoFromUtil(util),
      createdAt: Date.now(),
    };

    return network.customMove;
  };

  const isGoalReached = (network, util) => {
    const target = util && util.target;
    if (!target) return false;
    const goal = network.goal || { type: "point", x: 0, y: 0, spriteName: "" };

    if (goal.type === "sprite") {
      if (typeof target.isTouchingObject === "function") {
        return Boolean(target.isTouchingObject(goal.spriteName));
      }
      const other = findTargetByName(goal.spriteName, util);
      if (!other) return false;
      const dx = (target.x || 0) - (other.x || 0);
      const dy = (target.y || 0) - (other.y || 0);
      return Math.sqrt(dx * dx + dy * dy) < 20;
    }

    const point = goalPosition(network, util);
    const dx = (target.x || 0) - point.x;
    const dy = (target.y || 0) - point.y;
    return Math.sqrt(dx * dx + dy * dy) < 10;
  };

  const resetFailTimer = (network) => {
    network.fail = {
      ...(network.fail || {}),
      startedAt: Date.now(),
      failed: false,
      reason: "",
    };
  };

  const touchingFailSprite = (network, util) => {
    const fail = network.fail || {};
    const target = util && util.target;
    if (!target || !fail.spriteName) return false;

    if (typeof target.isTouchingObject === "function") {
      return Boolean(target.isTouchingObject(fail.spriteName));
    }

    const other = findTargetByName(fail.spriteName, util);
    if (!other) return false;
    const dx = (target.x || 0) - (other.x || 0);
    const dy = (target.y || 0) - (other.y || 0);
    return Math.sqrt(dx * dx + dy * dy) < 20;
  };

  const checkFailConditions = (network, util) => {
    const fail = network.fail || {};
    if (fail.failed) return true;

    if (fail.timeLimit > 0 && Date.now() - (fail.startedAt || Date.now()) > fail.timeLimit * 1000) {
      fail.failed = true;
      fail.reason = "time limit";
    }

    const target = util && util.target;
    if (!fail.failed && target && fail.spriteName) {
      const touching = touchingFailSprite(network, util);
      if (fail.recoveryUntilClear && touching) {
        network.fail = fail;
        return false;
      }
      if (fail.recoveryUntilClear && !touching) fail.recoveryUntilClear = false;
      fail.failed = touching;
      if (fail.failed) fail.reason = `touching ${fail.spriteName}`;
    }

    network.fail = fail;
    return Boolean(fail.failed);
  };

  const getStageCanvas = () => {
    if (typeof document === "undefined") return null;
    const canvases = Array.from(document.querySelectorAll("canvas"));
    if (!canvases.length) return null;
    return canvases.reduce((best, canvas) => {
      const rect = canvas.getBoundingClientRect();
      const bestRect = best.getBoundingClientRect();
      return rect.width * rect.height > bestRect.width * bestRect.height ? canvas : best;
    }, canvases[0]);
  };

  const stagePointToPage = (x, y, rect) => ({
    left: rect.left + ((x + STAGE_HALF_WIDTH) / (STAGE_HALF_WIDTH * 2)) * rect.width,
    top: rect.top + ((STAGE_HALF_HEIGHT - y) / (STAGE_HALF_HEIGHT * 2)) * rect.height,
  });

  const renderTrainingGhosts = () => {
    if (typeof document === "undefined") return;
    const visibleNetworks = Object.values(networks).filter(
      (network) => network.trainingGhosts && network.trainingGhosts.visible
    );

    if (!visibleNetworks.length) {
      if (trainingGhosts.layer) trainingGhosts.layer.replaceChildren();
      return;
    }

    const canvas = getStageCanvas();
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (!trainingGhosts.layer) {
      trainingGhosts.layer = document.createElement("div");
      trainingGhosts.layer.style.cssText =
        "position:fixed;left:0;top:0;right:0;bottom:0;z-index:30;pointer-events:none";
      document.body.appendChild(trainingGhosts.layer);
    }

    const dots = [];
    for (const network of visibleNetworks) {
      const settings = network.trainingGhosts || {};
      const limit = clampInteger(settings.limit, 1, 1000);
      const ghost = clampNumber(settings.ghost, 70, 0, 100);
      const opacity = Math.max(0.05, 1 - ghost / 100);
      const samples = network.samples.filter((sample) => sample.inputs && sample.inputs.length >= 4);
      const step = Math.max(1, Math.ceil(samples.length / limit));

      for (let i = 0; i < samples.length; i += step) {
        const sample = samples[i];
        const spriteX = clampNumber(sample.inputs[0], 0, -1, 1) * STAGE_HALF_WIDTH;
        const spriteY = clampNumber(sample.inputs[1], 0, -1, 1) * STAGE_HALF_HEIGHT;
        const outputX = sample.outputs && Number.isFinite(Number(sample.outputs[0])) ? sample.outputs[0] : 0;
        const outputY = sample.outputs && Number.isFinite(Number(sample.outputs[1])) ? sample.outputs[1] : 0;
        const point = stagePointToPage(spriteX, spriteY, rect);
        const angle = Math.atan2(-outputY, outputX);
        dots.push(
          `<span style="position:fixed;left:${point.left - 4}px;top:${
            point.top - 4
          }px;width:8px;height:8px;border-radius:50%;background:#37c29b;opacity:${opacity};box-shadow:0 0 0 1px rgba(0,0,0,.2)"></span>` +
            `<span style="position:fixed;left:${point.left}px;top:${
              point.top - 1
            }px;width:14px;height:2px;background:#37c29b;opacity:${opacity};transform-origin:left center;transform:rotate(${angle}rad)"></span>`
        );
      }
    }

    trainingGhosts.layer.innerHTML = dots.join("");
  };

  const renderPretrainView = () => {
    if (typeof document === "undefined") return;
    const visibleNetworks = Object.values(networks).filter(
      (network) => network.pretrain && network.pretrain.visible
    );

    if (!visibleNetworks.length) {
      if (pretrainView.layer) pretrainView.layer.replaceChildren();
      return;
    }

    const canvas = getStageCanvas();
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (!pretrainView.layer) {
      pretrainView.layer = document.createElement("div");
      pretrainView.layer.style.cssText =
        "position:fixed;left:0;top:0;right:0;bottom:0;z-index:31;pointer-events:none";
      document.body.appendChild(pretrainView.layer);
    }

    const pieces = [
      `<div style="position:fixed;left:${rect.left + 8}px;top:${
        rect.top + 8
      }px;max-width:${Math.max(
        180,
        rect.width - 16
      )}px;padding:7px 9px;border-radius:7px;background:rgba(17,24,39,.88);color:#fff2cc;font:700 12px system-ui,sans-serif;box-shadow:0 3px 12px rgba(0,0,0,.25)">Pretraining preview only - the real sprite is not moving here.</div>`,
    ];

    for (const network of visibleNetworks) {
      const paths = (network.pretrain && network.pretrain.paths) || [];
      paths.forEach((path, pathIndex) => {
        const points = Array.isArray(path.points) ? path.points : [];
        const opacity = pathIndex === paths.length - 1 ? 0.95 : 0.22;
        const color = path.success ? "#37c29b" : "#ffb020";

        for (let i = 1; i < points.length; i += 1) {
          const a = stagePointToPage(points[i - 1].x, points[i - 1].y, rect);
          const b = stagePointToPage(points[i].x, points[i].y, rect);
          const dx = b.left - a.left;
          const dy = b.top - a.top;
          const length = Math.sqrt(dx * dx + dy * dy);
          const angle = Math.atan2(dy, dx);
          pieces.push(
            `<span style="position:fixed;left:${a.left}px;top:${
              a.top
            }px;width:${length}px;height:3px;background:${color};opacity:${opacity};transform-origin:left center;transform:rotate(${angle}rad);border-radius:3px"></span>`
          );
        }

        const last = points[points.length - 1];
        if (last) {
          const end = stagePointToPage(last.x, last.y, rect);
          pieces.push(
            `<span style="position:fixed;left:${end.left - 5}px;top:${
              end.top - 5
            }px;width:10px;height:10px;border-radius:50%;background:${color};opacity:${opacity};box-shadow:0 0 0 1px rgba(0,0,0,.28)"></span>`
          );
        }
      });
    }

    pretrainView.layer.innerHTML = pieces.join("");
  };

  const shortNumber = (value) => Number(Number(value || 0).toFixed(3));

  const renderLayer = (values, label) => {
    const bars = values
      .slice(0, 32)
      .map((value) => {
        const amount = Math.min(1, Math.abs(Number(value) || 0));
        const color = value >= 0 ? "#37c29b" : "#ff6b7a";
        return `<span title="${shortNumber(value)}" style="display:inline-block;width:10px;height:${
          8 + amount * 44
        }px;background:${color};border-radius:2px;margin:0 2px;vertical-align:bottom"></span>`;
      })
      .join("");
    return `<div style="margin:10px 0"><div style="font-size:11px;color:#8b98b3;margin-bottom:4px">${label}</div><div style="height:56px;white-space:nowrap;overflow:hidden">${bars}</div></div>`;
  };

  const renderViewer = () => {
    if (!viewer.panel) return;
    const names = Object.keys(networks);
    if (!names.includes(viewer.selectedName)) viewer.selectedName = names[0] || DEFAULT_NAME;
    const network = networks[viewer.selectedName];

    if (!network) {
      viewer.panel.innerHTML = `<strong>No networks yet</strong><div style="margin-top:8px;color:#8b98b3">Run a neural net block first.</div>`;
      return;
    }

    const goal = network.goal || { type: "point", x: 0, y: 0, spriteName: "" };
    const goalText =
      goal.type === "sprite"
        ? `touching ${goal.spriteName || "sprite"}`
        : `x ${shortNumber(goal.x)}, y ${shortNumber(goal.y)}`;
    const fail = network.fail || {};
    const failText = fail.failed ? fail.reason || "failed" : "none";
    const move = network.customMove || {};
    const moveText = move.requested
      ? `direction ${shortNumber(move.desiredDirection)}, turn ${shortNumber(
          move.desiredTurn
        )}, vector ${shortNumber(move.desiredX)}, ${shortNumber(move.desiredY)}`
      : move.reason || "not requested";
    const ghostText =
      network.trainingGhosts && network.trainingGhosts.visible
        ? `showing ${network.trainingGhosts.limit} samples`
        : "hidden";
    const options = names
      .map(
        (name) =>
          `<option value="${name.replace(/"/g, "&quot;")}" ${
            name === viewer.selectedName ? "selected" : ""
          }>${name}</option>`
      )
      .join("");
    const layers = (network.lastActivations && network.lastActivations.length
      ? network.lastActivations
      : [[]]
    )
      .map((values, index) => renderLayer(values, index === 0 ? "inputs" : index === network.lastActivations.length - 1 ? "outputs" : `hidden ${index}`))
      .join("");

    viewer.panel.innerHTML = `
      <div style="display:flex;gap:8px;align-items:center;justify-content:space-between">
        <strong>Neural Thinking</strong>
        <button data-close="true" style="background:transparent;color:#c8d2ef;border:0;font-size:18px;cursor:pointer">x</button>
      </div>
      <select data-network-picker="true" style="width:100%;margin:10px 0 8px;padding:6px;border-radius:6px;background:#10182c;color:#f3f6ff;border:1px solid #34415f">${options}</select>
      <div style="font-size:12px;color:#c8d2ef;line-height:1.5">
        <div>shape: ${network.inputSize} -> ${network.hiddenSizes.join(" -> ")} -> ${network.outputSize}</div>
        <div>goal: ${goalText}</div>
        <div>fail: ${failText}</div>
        <div>custom move: ${moveText}</div>
        <div>training ghosts: ${ghostText}</div>
        <div>pretraining view: ${
          network.pretrain && network.pretrain.visible ? "showing preview only" : "hidden"
        }</div>
        <div>failure repairs: ${(network.recovery && network.recovery.count) || 0}</div>
        <div>loss: ${shortNumber(network.lastLoss)}</div>
        <div>prediction: ${formatNumberList(network.lastPrediction || []) || "none yet"}</div>
      </div>
      ${layers}
    `;

    const picker = viewer.panel.querySelector("[data-network-picker]");
    if (picker) {
      picker.addEventListener("change", () => {
        viewer.selectedName = picker.value;
        renderViewer();
      });
    }
    const close = viewer.panel.querySelector("[data-close]");
    if (close) close.addEventListener("click", toggleViewer);
  };

  function toggleViewer() {
    viewer.visible = !viewer.visible;
    if (viewer.panel) viewer.panel.style.display = viewer.visible ? "block" : "none";
    if (viewer.visible) renderViewer();
  }

  const installThinkingViewer = () => {
    if (viewer.installed || typeof document === "undefined") return;
    if (!document.body) {
      setTimeout(installThinkingViewer, 250);
      return;
    }
    const canvas = getStageCanvas();
    if (!canvas) {
      setTimeout(installThinkingViewer, 500);
      return;
    }

    viewer.installed = true;
    viewer.button = document.createElement("button");
    viewer.button.textContent = "Neural Thinking";
    viewer.button.title = "View the selected network's inputs, hidden layers, outputs, and goal";
    viewer.button.style.cssText =
      "position:fixed;z-index:40;padding:7px 10px;border-radius:8px;border:1px solid #2d6bff;background:#1b57e7;color:white;font:600 12px system-ui,sans-serif;box-shadow:0 6px 18px rgba(0,0,0,.25);cursor:pointer";
    viewer.button.addEventListener("click", toggleViewer);

    viewer.panel = document.createElement("div");
    viewer.panel.style.cssText =
      "display:none;position:fixed;width:320px;max-height:72vh;overflow:auto;z-index:41;padding:14px;border-radius:8px;border:1px solid #34415f;background:#0b1020;color:#f3f6ff;font:13px system-ui,sans-serif;box-shadow:0 18px 45px rgba(0,0,0,.35)";

    document.body.appendChild(viewer.button);
    document.body.appendChild(viewer.panel);

    const placeViewer = () => {
      const stageCanvas = getStageCanvas();
      if (!stageCanvas || !viewer.button || !viewer.panel) return;
      const rect = stageCanvas.getBoundingClientRect();
      const buttonTop = Math.max(8, rect.top + 8);
      const buttonLeft = Math.max(8, rect.right - viewer.button.offsetWidth - 8);
      viewer.button.style.left = `${buttonLeft}px`;
      viewer.button.style.top = `${buttonTop}px`;
      viewer.panel.style.left = `${Math.max(8, rect.right - 328)}px`;
      viewer.panel.style.top = `${buttonTop + 38}px`;
    };

    placeViewer();
    window.addEventListener("resize", placeViewer);
    viewer.timer = setInterval(() => {
      placeViewer();
      if (viewer.visible) renderViewer();
      renderTrainingGhosts();
      renderPretrainView();
    }, 500);
    trainingGhosts.timer = viewer.timer;
  };

  class NeuralNetworkExtension {
    constructor() {
      installThinkingViewer();
    }

    getInfo() {
      return {
        id: "scratchNeurals",
        name: "Neural Nets",
        color1: "#3f7cff",
        color2: "#2f5fd2",
        color3: "#2448a6",
        blocks: [
          {
            opcode: "makeNetwork",
            blockType: Scratch.BlockType.COMMAND,
            text: "make neural net [NAME] inputs [INPUTS] hidden [HIDDEN] outputs [OUTPUTS] rate [RATE]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
              INPUTS: { type: Scratch.ArgumentType.NUMBER, defaultValue: 1 },
              HIDDEN: { type: Scratch.ArgumentType.STRING, defaultValue: "8" },
              OUTPUTS: { type: Scratch.ArgumentType.NUMBER, defaultValue: 1 },
              RATE: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.03 },
            },
          },
          {
            opcode: "makeSpriteController",
            blockType: Scratch.BlockType.COMMAND,
            text: "make sprite controller [NAME] hidden [HIDDEN] rate [RATE]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
              HIDDEN: { type: Scratch.ArgumentType.STRING, defaultValue: "12,8" },
              RATE: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.04 },
            },
          },
          {
            opcode: "setActivation",
            blockType: Scratch.BlockType.COMMAND,
            text: "set [NAME] hidden activation to [ACTIVATION]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
              ACTIVATION: {
                type: Scratch.ArgumentType.STRING,
                menu: "activations",
                defaultValue: "tanh",
              },
            },
          },
          {
            opcode: "setLearningRate",
            blockType: Scratch.BlockType.COMMAND,
            text: "set [NAME] learning rate to [RATE]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
              RATE: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.03 },
            },
          },
          "---",
          {
            opcode: "setGoalPoint",
            blockType: Scratch.BlockType.COMMAND,
            text: "set [NAME] goal to x [X] y [Y]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
              X: { type: Scratch.ArgumentType.NUMBER, defaultValue: 100 },
              Y: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0 },
            },
          },
          {
            opcode: "setGoalSprite",
            blockType: Scratch.BlockType.COMMAND,
            text: "set [NAME] goal to touching sprite [SPRITE]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
              SPRITE: {
                type: Scratch.ArgumentType.STRING,
                defaultValue: "Sprite1",
              },
            },
          },
          {
            opcode: "goalReached",
            blockType: Scratch.BlockType.BOOLEAN || Scratch.BlockType.REPORTER,
            text: "[NAME] goal reached?",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
            },
          },
          "---",
          {
            opcode: "addTrainingPair",
            blockType: Scratch.BlockType.COMMAND,
            text: "add training pair to [NAME] inputs [INPUTS] outputs [OUTPUTS]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
              INPUTS: { type: Scratch.ArgumentType.STRING, defaultValue: "0.5" },
              OUTPUTS: { type: Scratch.ArgumentType.STRING, defaultValue: "0.25" },
            },
          },
          {
            opcode: "clearTrainingData",
            blockType: Scratch.BlockType.COMMAND,
            text: "clear training data for [NAME]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
            },
          },
          {
            opcode: "trainingCount",
            blockType: Scratch.BlockType.REPORTER,
            text: "training pair count for [NAME]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
            },
          },
          {
            opcode: "trainingOwnerId",
            blockType: Scratch.BlockType.REPORTER,
            text: "training id for this sprite/clone",
          },
          {
            opcode: "ownedTrainingCount",
            blockType: Scratch.BlockType.REPORTER,
            text: "training pair count for [NAME] from [OWNER]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
              OWNER: { type: Scratch.ArgumentType.STRING, defaultValue: "this" },
            },
          },
          {
            opcode: "exportTrainingForOwner",
            blockType: Scratch.BlockType.REPORTER,
            text: "export [NAME] training from [OWNER]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
              OWNER: { type: Scratch.ArgumentType.STRING, defaultValue: "this" },
            },
          },
          {
            opcode: "importTrainingPacketBlock",
            blockType: Scratch.BlockType.COMMAND,
            text: "import training into [NAME] from JSON [JSON]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
              JSON: { type: Scratch.ArgumentType.STRING, defaultValue: "{}" },
            },
          },
          {
            opcode: "lastTrainingImportInfo",
            blockType: Scratch.BlockType.REPORTER,
            text: "last training import info for [NAME]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
            },
          },
          {
            opcode: "train",
            blockType: Scratch.BlockType.COMMAND,
            text: "train [NAME] for [EPOCHS] epochs",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
              EPOCHS: { type: Scratch.ArgumentType.NUMBER, defaultValue: 100 },
            },
          },
          {
            opcode: "trainOne",
            blockType: Scratch.BlockType.COMMAND,
            text: "train [NAME] once on inputs [INPUTS] outputs [OUTPUTS]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
              INPUTS: { type: Scratch.ArgumentType.STRING, defaultValue: "0.5" },
              OUTPUTS: { type: Scratch.ArgumentType.STRING, defaultValue: "0.25" },
            },
          },
          {
            opcode: "addFunctionSamplesBlock",
            blockType: Scratch.BlockType.COMMAND,
            text: "add function samples to [NAME] y = [FORMULA] x from [START] to [END] samples [COUNT]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
              FORMULA: { type: Scratch.ArgumentType.STRING, defaultValue: "x * x" },
              START: { type: Scratch.ArgumentType.NUMBER, defaultValue: -1 },
              END: { type: Scratch.ArgumentType.NUMBER, defaultValue: 1 },
              COUNT: { type: Scratch.ArgumentType.NUMBER, defaultValue: 21 },
            },
          },
          {
            opcode: "trainFunction",
            blockType: Scratch.BlockType.COMMAND,
            text: "train [NAME] to y = [FORMULA] x from [START] to [END] samples [COUNT] epochs [EPOCHS]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
              FORMULA: { type: Scratch.ArgumentType.STRING, defaultValue: "x * x" },
              START: { type: Scratch.ArgumentType.NUMBER, defaultValue: -1 },
              END: { type: Scratch.ArgumentType.NUMBER, defaultValue: 1 },
              COUNT: { type: Scratch.ArgumentType.NUMBER, defaultValue: 21 },
              EPOCHS: { type: Scratch.ArgumentType.NUMBER, defaultValue: 1000 },
            },
          },
          {
            opcode: "trainController",
            blockType: Scratch.BlockType.COMMAND,
            text: "train [NAME] controller with [SAMPLES] goal samples for [EPOCHS] epochs",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
              SAMPLES: { type: Scratch.ArgumentType.NUMBER, defaultValue: 100 },
              EPOCHS: { type: Scratch.ArgumentType.NUMBER, defaultValue: 20 },
            },
          },
          {
            opcode: "pretrainController",
            blockType: Scratch.BlockType.COMMAND,
            text: "pretrain [NAME] from this sprite rounds [ROUNDS] samples [SAMPLES] epochs [EPOCHS]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
              ROUNDS: { type: Scratch.ArgumentType.NUMBER, defaultValue: 20 },
              SAMPLES: { type: Scratch.ArgumentType.NUMBER, defaultValue: 120 },
              EPOCHS: { type: Scratch.ArgumentType.NUMBER, defaultValue: 20 },
            },
          },
          {
            opcode: "pretrainSucceeded",
            blockType: Scratch.BlockType.BOOLEAN || Scratch.BlockType.REPORTER,
            text: "[NAME] pretrain succeeded?",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
            },
          },
          {
            opcode: "pretrainInfo",
            blockType: Scratch.BlockType.REPORTER,
            text: "pretrain info for [NAME]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
            },
          },
          {
            opcode: "controlSprite",
            blockType: Scratch.BlockType.COMMAND,
            text: "control this sprite with [NAME] speed [SPEED] train [TRAIN] samples",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
              SPEED: { type: Scratch.ArgumentType.NUMBER, defaultValue: 5 },
              TRAIN: { type: Scratch.ArgumentType.NUMBER, defaultValue: 2 },
            },
          },
          {
            opcode: "steerSprite",
            blockType: Scratch.BlockType.COMMAND,
            text: "steer this sprite with [NAME] speed [SPEED] turn [TURN] train [TRAIN] samples",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
              SPEED: { type: Scratch.ArgumentType.NUMBER, defaultValue: 4 },
              TURN: { type: Scratch.ArgumentType.NUMBER, defaultValue: 12 },
              TRAIN: { type: Scratch.ArgumentType.NUMBER, defaultValue: 2 },
            },
          },
          {
            opcode: "withCustomMovement",
            blockType: Scratch.BlockType.CONDITIONAL || Scratch.BlockType.COMMAND,
            branchCount: 1,
            text: "with [NAME] custom movement speed [SPEED] turn [TURN] train [TRAIN] samples",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
              SPEED: { type: Scratch.ArgumentType.NUMBER, defaultValue: 4 },
              TURN: { type: Scratch.ArgumentType.NUMBER, defaultValue: 12 },
              TRAIN: { type: Scratch.ArgumentType.NUMBER, defaultValue: 2 },
            },
          },
          {
            opcode: "chooseCustomMove",
            blockType: Scratch.BlockType.COMMAND,
            text: "ask [NAME] for custom move speed [SPEED] turn [TURN] train [TRAIN] samples",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
              SPEED: { type: Scratch.ArgumentType.NUMBER, defaultValue: 4 },
              TURN: { type: Scratch.ArgumentType.NUMBER, defaultValue: 12 },
              TRAIN: { type: Scratch.ArgumentType.NUMBER, defaultValue: 2 },
            },
          },
          {
            opcode: "customMoveReady",
            blockType: Scratch.BlockType.BOOLEAN || Scratch.BlockType.REPORTER,
            text: "[NAME] custom move ready?",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
            },
          },
          {
            opcode: "customMoveDirection",
            blockType: Scratch.BlockType.REPORTER,
            text: "[NAME] custom move direction",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
            },
          },
          {
            opcode: "customMoveTurn",
            blockType: Scratch.BlockType.REPORTER,
            text: "[NAME] custom move turn",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
            },
          },
          {
            opcode: "customMoveX",
            blockType: Scratch.BlockType.REPORTER,
            text: "[NAME] custom move x",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
            },
          },
          {
            opcode: "customMoveY",
            blockType: Scratch.BlockType.REPORTER,
            text: "[NAME] custom move y",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
            },
          },
          {
            opcode: "customMoveSpeed",
            blockType: Scratch.BlockType.REPORTER,
            text: "[NAME] custom move speed",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
            },
          },
          {
            opcode: "customMoveReason",
            blockType: Scratch.BlockType.REPORTER,
            text: "[NAME] custom move reason",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
            },
          },
          {
            opcode: "showTrainingGhosts",
            blockType: Scratch.BlockType.COMMAND,
            text: "[SHOW] training ghosts for [NAME] limit [LIMIT] ghost [GHOST]",
            arguments: {
              SHOW: {
                type: Scratch.ArgumentType.STRING,
                menu: "showHide",
                defaultValue: "show",
              },
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
              LIMIT: { type: Scratch.ArgumentType.NUMBER, defaultValue: 150 },
              GHOST: { type: Scratch.ArgumentType.NUMBER, defaultValue: 70 },
            },
          },
          {
            opcode: "showPretrainView",
            blockType: Scratch.BlockType.COMMAND,
            text: "[SHOW] pretraining view for [NAME]",
            arguments: {
              SHOW: {
                type: Scratch.ArgumentType.STRING,
                menu: "showHide",
                defaultValue: "show",
              },
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
            },
          },
          "---",
          {
            opcode: "failIfTouchingSprite",
            blockType: Scratch.BlockType.COMMAND,
            text: "set [NAME] fail if touching sprite [SPRITE]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
              SPRITE: { type: Scratch.ArgumentType.STRING, defaultValue: "Danger" },
            },
          },
          {
            opcode: "setTimeLimit",
            blockType: Scratch.BlockType.COMMAND,
            text: "set [NAME] time limit to [SECONDS] seconds",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
              SECONDS: { type: Scratch.ArgumentType.NUMBER, defaultValue: 10 },
            },
          },
          {
            opcode: "resetRun",
            blockType: Scratch.BlockType.COMMAND,
            text: "reset [NAME] run timer and fail state",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
            },
          },
          {
            opcode: "failed",
            blockType: Scratch.BlockType.BOOLEAN || Scratch.BlockType.REPORTER,
            text: "[NAME] failed?",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
            },
          },
          {
            opcode: "failReason",
            blockType: Scratch.BlockType.REPORTER,
            text: "fail reason for [NAME]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
            },
          },
          {
            opcode: "learnFromFailureBlock",
            blockType: Scratch.BlockType.COMMAND,
            text: "learn from [NAME] failure samples [SAMPLES] epochs [EPOCHS]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
              SAMPLES: { type: Scratch.ArgumentType.NUMBER, defaultValue: 120 },
              EPOCHS: { type: Scratch.ArgumentType.NUMBER, defaultValue: 20 },
            },
          },
          {
            opcode: "failureLearningInfo",
            blockType: Scratch.BlockType.REPORTER,
            text: "failure learning info for [NAME]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: "controller" },
            },
          },
          {
            opcode: "loss",
            blockType: Scratch.BlockType.REPORTER,
            text: "last loss for [NAME]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
            },
          },
          "---",
          {
            opcode: "predictOutput",
            blockType: Scratch.BlockType.REPORTER,
            text: "prediction from [NAME] inputs [INPUTS] output [INDEX]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
              INPUTS: { type: Scratch.ArgumentType.STRING, defaultValue: "0.5" },
              INDEX: { type: Scratch.ArgumentType.NUMBER, defaultValue: 1 },
            },
          },
          {
            opcode: "predictList",
            blockType: Scratch.BlockType.REPORTER,
            text: "prediction list from [NAME] inputs [INPUTS]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
              INPUTS: { type: Scratch.ArgumentType.STRING, defaultValue: "0.5" },
            },
          },
          "---",
          {
            opcode: "randomize",
            blockType: Scratch.BlockType.COMMAND,
            text: "randomize [NAME]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
            },
          },
          {
            opcode: "exportNetwork",
            blockType: Scratch.BlockType.REPORTER,
            text: "export [NAME] as JSON",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
            },
          },
          {
            opcode: "importNetwork",
            blockType: Scratch.BlockType.COMMAND,
            text: "import [NAME] from JSON [JSON]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
              JSON: { type: Scratch.ArgumentType.STRING, defaultValue: "{}" },
            },
          },
          {
            opcode: "networkInfo",
            blockType: Scratch.BlockType.REPORTER,
            text: "info for [NAME]",
            arguments: {
              NAME: { type: Scratch.ArgumentType.STRING, defaultValue: DEFAULT_NAME },
            },
          },
        ],
        menus: {
          activations: {
            acceptReporters: true,
            items: ["tanh", "relu", "sigmoid", "linear"],
          },
          showHide: {
            acceptReporters: true,
            items: ["show", "hide"],
          },
        },
      };
    }

    makeNetwork(args) {
      const name = cleanName(args.NAME);
      const inputSize = clampInteger(args.INPUTS, 1, 32);
      const hiddenSizes = parseHiddenLayers(args.HIDDEN);
      const outputSize = clampInteger(args.OUTPUTS, 1, 32);
      const learningRate = clampNumber(args.RATE, 0.03, 0.000001, 1);
      networks[name] = makeNetwork(inputSize, hiddenSizes, outputSize, learningRate, "tanh");
    }

    makeSpriteController(args) {
      const name = cleanName(args.NAME);
      const hiddenSizes = parseHiddenLayers(args.HIDDEN);
      const learningRate = clampNumber(args.RATE, 0.04, 0.000001, 1);
      networks[name] = makeNetwork(4, hiddenSizes, 2, learningRate, "tanh");
      networks[name].goal = { type: "point", x: 0, y: 0, spriteName: "" };
    }

    setActivation(args) {
      const network = ensureNetwork(args.NAME);
      const activation = String(args.ACTIVATION || "tanh").toLowerCase();
      network.activation = ["tanh", "relu", "sigmoid", "linear"].includes(activation)
        ? activation
        : "tanh";
    }

    setLearningRate(args) {
      const network = ensureNetwork(args.NAME);
      network.learningRate = clampNumber(args.RATE, network.learningRate, 0.000001, 1);
    }

    setGoalPoint(args) {
      const network = ensureControllerNetwork(args.NAME);
      network.goal = {
        type: "point",
        x: clampNumber(args.X, 0, -STAGE_HALF_WIDTH, STAGE_HALF_WIDTH),
        y: clampNumber(args.Y, 0, -STAGE_HALF_HEIGHT, STAGE_HALF_HEIGHT),
        spriteName: "",
      };
    }

    setGoalSprite(args) {
      const network = ensureControllerNetwork(args.NAME);
      network.goal = {
        type: "sprite",
        x: 0,
        y: 0,
        spriteName: String(args.SPRITE || "").trim(),
      };
    }

    goalReached(args, util) {
      return isGoalReached(ensureControllerNetwork(args.NAME), util);
    }

    addTrainingPair(args, util) {
      const network = ensureNetwork(args.NAME);
      network.samples.push(
        ownedSample(
          network,
          parseNumberList(args.INPUTS),
          parseNumberList(args.OUTPUTS),
          ownerInfoFromUtil(util)
        )
      );
    }

    clearTrainingData(args) {
      const network = ensureNetwork(args.NAME);
      network.samples = [];
      network.lastLoss = 0;
    }

    trainingCount(args) {
      return ensureNetwork(args.NAME).samples.length;
    }

    trainingOwnerId(_args, util) {
      return ownerInfoFromUtil(util).key;
    }

    ownedTrainingCount(args, util) {
      const network = ensureNetwork(args.NAME);
      const owner = ownerQuery(args.OWNER, util);
      return network.samples.filter((sample) => sampleMatchesOwner(sample, owner)).length;
    }

    exportTrainingForOwner(args, util) {
      const network = ensureNetwork(args.NAME);
      return exportTrainingPacket(network, args.NAME, ownerQuery(args.OWNER, util));
    }

    importTrainingPacketBlock(args, util) {
      importTrainingPacket(ensureNetwork(args.NAME), args.JSON, util);
    }

    lastTrainingImportInfo(args) {
      const network = ensureNetwork(args.NAME);
      const info = network.lastImport || { count: 0, from: "none", importedBy: "" };
      return `${info.count || 0} samples from ${info.from || "none"}`;
    }

    train(args) {
      const network = ensureNetwork(args.NAME);
      const epochs = clampInteger(args.EPOCHS, 1, 100000);
      for (let epoch = 0; epoch < epochs; epoch += 1) {
        trainEpoch(network);
      }
    }

    trainOne(args) {
      const network = ensureNetwork(args.NAME);
      network.lastLoss = trainSample(
        network,
        parseNumberList(args.INPUTS),
        parseNumberList(args.OUTPUTS)
      );
    }

    addFunctionSamplesBlock(args, util) {
      addFunctionSamples(
        ensureNetwork(args.NAME),
        args.FORMULA,
        args.START,
        args.END,
        args.COUNT,
        ownerInfoFromUtil(util)
      );
    }

    trainFunction(args, util) {
      const network = ensureNetwork(args.NAME);
      addFunctionSamples(
        network,
        args.FORMULA,
        args.START,
        args.END,
        args.COUNT,
        ownerInfoFromUtil(util)
      );
      const epochs = clampInteger(args.EPOCHS, 1, 100000);
      for (let epoch = 0; epoch < epochs; epoch += 1) {
        trainEpoch(network);
      }
    }

    trainController(args, util) {
      const network = ensureControllerNetwork(args.NAME);
      addControllerSamples(network, args.SAMPLES, util);
      const epochs = clampInteger(args.EPOCHS, 1, 100000);
      for (let epoch = 0; epoch < epochs; epoch += 1) {
        trainEpoch(network);
      }
    }

    pretrainController(args, util) {
      pretrainControllerFromSprite(ensureControllerNetwork(args.NAME), args, util);
    }

    pretrainSucceeded(args) {
      const network = ensureControllerNetwork(args.NAME);
      return Boolean(network.pretrain && network.pretrain.success);
    }

    pretrainInfo(args) {
      const network = ensureControllerNetwork(args.NAME);
      const pretrain = network.pretrain || {
        success: false,
        rounds: 0,
        reason: "not run",
        steps: 0,
      };
      return [
        pretrain.success ? "success" : "not ready",
        `rounds ${pretrain.rounds || 0}`,
        `steps ${pretrain.steps || 0}`,
        pretrain.reason || "not run",
      ].join(" | ");
    }

    controlSprite(args, util) {
      runSteeringController(args, util, 12);
    }

    steerSprite(args, util) {
      runSteeringController(args, util, 12);
    }

    withCustomMovement(args, util) {
      const move = computeCustomMoveIntent(args, util, 12);
      return Boolean(move && move.requested);
    }

    chooseCustomMove(args, util) {
      computeCustomMoveIntent(args, util, 12);
    }

    customMoveReady(args) {
      const move = ensureControllerNetwork(args.NAME).customMove || {};
      return Boolean(move.requested);
    }

    customMoveDirection(args) {
      const move = ensureControllerNetwork(args.NAME).customMove || {};
      return Number.isFinite(Number(move.desiredDirection)) ? Number(move.desiredDirection) : 90;
    }

    customMoveTurn(args) {
      const move = ensureControllerNetwork(args.NAME).customMove || {};
      return Number.isFinite(Number(move.desiredTurn)) ? Number(move.desiredTurn) : 0;
    }

    customMoveX(args) {
      const move = ensureControllerNetwork(args.NAME).customMove || {};
      return Number.isFinite(Number(move.desiredX)) ? Number(move.desiredX) : 0;
    }

    customMoveY(args) {
      const move = ensureControllerNetwork(args.NAME).customMove || {};
      return Number.isFinite(Number(move.desiredY)) ? Number(move.desiredY) : 0;
    }

    customMoveSpeed(args) {
      const move = ensureControllerNetwork(args.NAME).customMove || {};
      return Number.isFinite(Number(move.speed)) ? Number(move.speed) : 0;
    }

    customMoveReason(args) {
      const move = ensureControllerNetwork(args.NAME).customMove || {};
      return move.reason || (move.requested ? "move" : "not requested");
    }

    showTrainingGhosts(args) {
      const network = ensureControllerNetwork(args.NAME);
      const show = String(args.SHOW || "show").toLowerCase() !== "hide";
      network.trainingGhosts = {
        visible: show,
        limit: clampInteger(args.LIMIT, 1, 1000),
        ghost: clampNumber(args.GHOST, 70, 0, 100),
      };
      renderTrainingGhosts();
    }

    showPretrainView(args) {
      const network = ensureControllerNetwork(args.NAME);
      network.pretrain = {
        ...(network.pretrain || {}),
        visible: String(args.SHOW || "show").toLowerCase() !== "hide",
      };
      renderPretrainView();
    }

    failIfTouchingSprite(args) {
      const network = ensureControllerNetwork(args.NAME);
      network.fail = {
        ...(network.fail || {}),
        spriteName: String(args.SPRITE || "").trim(),
      };
    }

    setTimeLimit(args) {
      const network = ensureControllerNetwork(args.NAME);
      network.fail = {
        ...(network.fail || {}),
        timeLimit: clampNumber(args.SECONDS, 0, 0, 999999),
      };
      resetFailTimer(network);
    }

    resetRun(args) {
      resetFailTimer(ensureControllerNetwork(args.NAME));
    }

    failed(args, util) {
      return checkFailConditions(ensureControllerNetwork(args.NAME), util);
    }

    failReason(args) {
      const network = ensureControllerNetwork(args.NAME);
      return (network.fail && network.fail.reason) || "";
    }

    learnFromFailureBlock(args, util) {
      learnFromFailure(ensureControllerNetwork(args.NAME), args, util);
    }

    failureLearningInfo(args) {
      const network = ensureControllerNetwork(args.NAME);
      const recovery = network.recovery || {
        count: 0,
        reason: "not run",
        samples: 0,
        epochs: 0,
      };
      return [
        `repairs ${recovery.count || 0}`,
        recovery.reason || "not run",
        `${recovery.samples || 0} samples`,
        `${recovery.epochs || 0} epochs`,
      ].join(" | ");
    }

    loss(args) {
      return ensureNetwork(args.NAME).lastLoss;
    }

    predictOutput(args) {
      const network = ensureNetwork(args.NAME);
      const activations = forward(network, parseNumberList(args.INPUTS));
      const prediction = activations[activations.length - 1];
      const index = clampInteger(args.INDEX, 1, network.outputSize) - 1;
      return prediction[index];
    }

    predictList(args) {
      const network = ensureNetwork(args.NAME);
      const activations = forward(network, parseNumberList(args.INPUTS));
      return formatNumberList(activations[activations.length - 1]);
    }

    randomize(args) {
      const name = cleanName(args.NAME);
      const network = ensureNetwork(name);
      networks[name] = makeNetwork(
        network.inputSize,
        network.hiddenSizes,
        network.outputSize,
        network.learningRate,
        network.activation
      );
      networks[name].samples = network.samples;
      networks[name].goal = network.goal;
      networks[name].fail = network.fail;
      networks[name].trainingGhosts = network.trainingGhosts;
      networks[name].pretrain = network.pretrain;
      networks[name].recovery = network.recovery;
      networks[name].customMove = network.customMove;
    }

    exportNetwork(args) {
      const network = ensureNetwork(args.NAME);
      return JSON.stringify(network);
    }

    importNetwork(args) {
      try {
        const imported = JSON.parse(String(args.JSON || "{}"));
        if (
          !imported ||
          !Array.isArray(imported.layers) ||
          !Number.isFinite(Number(imported.inputSize)) ||
          !Number.isFinite(Number(imported.outputSize))
        ) {
          return;
        }

        const name = cleanName(args.NAME);
        networks[name] = {
          inputSize: clampInteger(imported.inputSize, 1, 32),
          hiddenSizes: Array.isArray(imported.hiddenSizes)
            ? imported.hiddenSizes.map((size) => clampInteger(size, 1, 128))
            : [8],
          outputSize: clampInteger(imported.outputSize, 1, 32),
          learningRate: clampNumber(imported.learningRate, 0.03, 0.000001, 1),
          activation: ["tanh", "relu", "sigmoid", "linear"].includes(imported.activation)
            ? imported.activation
            : "tanh",
          layers: imported.layers,
          samples: Array.isArray(imported.samples) ? imported.samples : [],
          goal: imported.goal || { type: "point", x: 0, y: 0, spriteName: "" },
          fail: imported.fail || {
            spriteName: "",
            timeLimit: 0,
            startedAt: Date.now(),
            failed: false,
            reason: "",
            recoveryUntilClear: false,
          },
          trainingGhosts: imported.trainingGhosts || {
            visible: false,
            limit: 150,
            ghost: 70,
          },
          pretrain: imported.pretrain || {
            success: false,
            rounds: 0,
            reason: "not run",
            steps: 0,
          },
          recovery: imported.recovery || {
            count: 0,
            reason: "not run",
            samples: 0,
            epochs: 0,
          },
          customMove: imported.customMove || {
            requested: false,
            reason: "not requested",
            desiredDirection: 90,
            desiredTurn: 0,
            desiredX: 0,
            desiredY: 0,
            speed: 4,
            train: 0,
          },
          lastInputs: Array.isArray(imported.lastInputs) ? imported.lastInputs : [],
          lastPrediction: Array.isArray(imported.lastPrediction) ? imported.lastPrediction : [],
          lastActivations: Array.isArray(imported.lastActivations) ? imported.lastActivations : [],
          lastLoss: Number(imported.lastLoss) || 0,
        };
      } catch (_error) {
        // Invalid JSON should simply leave the current project running.
      }
    }

    networkInfo(args) {
      const network = ensureNetwork(args.NAME);
      return [
        `${network.inputSize} inputs`,
        `hidden ${network.hiddenSizes.join(",")}`,
        `${network.outputSize} outputs`,
        `rate ${network.learningRate}`,
        `${network.samples.length} pairs`,
        `pretrain ${(network.pretrain && network.pretrain.reason) || "not run"}${
          network.pretrain && network.pretrain.visible ? " preview" : ""
        }`,
        `repairs ${(network.recovery && network.recovery.count) || 0}`,
        `loss ${Number(network.lastLoss.toFixed(6))}`,
      ].join(" | ");
    }
  }

  Scratch.extensions.register(new NeuralNetworkExtension());
})(Scratch);
