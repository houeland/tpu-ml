import { ChangeEvent, PropsWithChildren, useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import * as Plot from "@observablehq/plot";

type Result = {
  peak_lr_value: number;
  lr_decay: number;
  used_nesterov_momentum: boolean;
  epoch_100_test_accuracy: number;
};

type Row = {
  peak_lr_value: number;
  lr_decay: number;
  diff_using_nesterov: number;
  diff_bucketized: string;
  n0: number;
  n1: number;
  n05: number;
  nmax: number;
};

function checkExists<T>(v: NonNullable<T> | null | undefined): T {
  if (v === null || v === undefined) {
    throw new Error(`checkExists(): received input ${v}`);
  } else {
    return v;
  }
}

function bucketize(diff: number) {
  if (diff < -0.5) {
    return "-0.5+";
  } else if (diff < -0.1) {
    return "-0.5 to -0.1";
  } else if (diff < -0.01) {
    return "-0.1 to -0.01";
  } else if (diff < 0.01) {
    return "-0.01 to +0.01";
  } else if (diff < 0.1) {
    return "+0.01 to +0.1";
  } else if (diff < 0.5) {
    return "+0.1 to +0.5";
  } else {
    return "+0.5+";
  }
}

type Options = {
  bin_width: "tiny" | "small" | "medium" | "large" | "huge";
  vistype: "lr_heatmap" | "nesterov_vs_max_accuracy" | "other";
  min_value?: number;
  which_accuracy: "only_nesterov" | "never_nesterov" | "best" | "worst" | "diff_nesterov";
  aggregation_style: "hex" | "rect";
  skip_divergent_regions: boolean;
};

function processData(data: Result[], options: Options) {
  const with_nesterov = new Map<string, number>();
  const without_nesterov = new Map<string, number>();
  const key_lookup = new Map<string, { peak_lr_value: number; lr_decay: number }>();
  for (const d of data) {
    const key = `peak_lr=${d.peak_lr_value} lr_decay=${d.lr_decay}`;
    key_lookup.set(key, d);
    if (d.used_nesterov_momentum) {
      with_nesterov.set(key, d.epoch_100_test_accuracy);
    } else {
      without_nesterov.set(key, d.epoch_100_test_accuracy);
    }
  }

  let rows: Row[] = [];
  for (const [key, value] of key_lookup.entries()) {
    const n0 = without_nesterov.get(key);
    const n1 = with_nesterov.get(key);
    if (n0 !== undefined && n1 !== undefined) {
      const diff_using_nesterov = n1 - n0;
      rows.push({
        ...value,
        diff_using_nesterov,
        n0,
        n1,
        n05: (n0 + n1) / 2,
        nmax: Math.max(n0, n1),
        diff_bucketized: bucketize(diff_using_nesterov),
      });
    }
  }
  if (options.min_value !== undefined) {
    const min = options.min_value;
    switch (options.which_accuracy) {
      case "best":
        rows = rows.filter((d) => d.nmax >= min);
        break;
      case "never_nesterov":
        rows = rows.filter((d) => d.n0 >= min);
        break;
      case "only_nesterov":
        rows = rows.filter((d) => d.n1 >= min);
        break;
      case "worst":
      case "diff_nesterov":
        rows = rows.filter((d) => d.n0 + d.n1 - d.nmax >= min);
        break;
    }
  }
  // if (options.skip_divergent_regions) {
  //   rows = rows.filter((d) => d.nmax - Math.min(d.n0, d.n1) < 0.1);
  // }
  return rows;
}

function useCheckbox(defaultValue: boolean) {
  const [value, setValue] = useState(defaultValue);
  const onChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      setValue(event.target.checked);
    },
    [setValue]
  );
  return { value, onChange };
}

function useSelect(defaultValue: string) {
  const [value, setValue] = useState(defaultValue);
  const onChange = useCallback(
    (event: ChangeEvent<HTMLSelectElement>) => {
      setValue(event.target.value);
    },
    [setValue]
  );
  return { value, onChange };
}

function color_by_nesterov_diff_sum(rows: Row[]) {
  let nesterov_diff_sum = 0;
  for (const r of rows) {
    nesterov_diff_sum += r.diff_using_nesterov;
  }
  return nesterov_diff_sum / rows.length;
}

function color_by_average_test_accuracy(rows: Row[], options: Options) {
  let accuracy_sum = 0;
  for (const r of rows) {
    switch (options.which_accuracy) {
      case "best":
        accuracy_sum += r.nmax;
        break;
      case "never_nesterov":
        accuracy_sum += r.n0;
        break;
      case "only_nesterov":
        accuracy_sum += r.n1;
        break;
      case "worst":
        accuracy_sum += r.n0;
        accuracy_sum += r.n1;
        accuracy_sum -= r.nmax;
        break;
      case "diff_nesterov":
        return color_by_nesterov_diff_sum(rows);
    }
  }
  if (options.skip_divergent_regions) {
    let worst = checkExists(rows[0]).nmax;
    let best = checkExists(rows[0]).n0;
    for (const r of rows) {
      worst = Math.min(worst, r.n0, r.n1);
      best = Math.max(best, r.n0, r.n1);
    }
    if (best - worst > 0.1) {
      return undefined;
    }
  }
  return accuracy_sum / rows.length;
}

export function App() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<Result[]>();
  const vistype = useSelect("lr_heatmap");
  const coloring_by = useSelect("best");
  const filter_nmax = useSelect("0.8");
  const hex_aggregation = useCheckbox(true);
  const divergent = useCheckbox(false);
  const binning = useSelect("medium");

  useEffect(() => {
    async function load() {
      const response = await fetch("./cifar10-results-compressed.json");
      const data = (await response.json()).map(([peak_lr_value, lr_decay, nesterov, epoch_100_test_accuracy]: [number, number, number, number]) => ({
        peak_lr_value,
        lr_decay,
        used_nesterov_momentum: nesterov === 1,
        epoch_100_test_accuracy,
      })) as Result[];
      setData(data);
    }
    load();
  }, []);

  const options = useMemo(() => {
    return {
      vistype: vistype.value,
      bin_width: binning.value,
      min_value: Number(filter_nmax.value),
      which_accuracy: coloring_by.value,
      aggregation_style: hex_aggregation.value ? "hex" : "rect",
      skip_divergent_regions: divergent.value,
    } as Options;
  }, [vistype.value, coloring_by.value, hex_aggregation.value, divergent.value, binning.value, filter_nmax.value]);

  useEffect(() => {
    if (data === undefined) return;
    const values = processData(data, options);
    let outerDot;
    let innerBin;
    switch (options.aggregation_style) {
      case "hex":
        outerDot = Plot.dot;
        innerBin = Plot.hexbin;
        break;
      case "rect":
        outerDot = Plot.rect;
        innerBin = Plot.bin;
        break;
    }
    let binWidth;
    let thresholds;
    switch (options.bin_width) {
      case "tiny":
        binWidth = 2;
        thresholds = 150;
        break;
      case "small":
        binWidth = 5;
        thresholds = 100;
        break;
      case "medium":
        binWidth = 10;
        thresholds = 50;
        break;
      case "large":
        binWidth = 20;
        thresholds = 20;
        break;
      case "huge":
        binWidth = 40;
        thresholds = 10;
        break;
    }
    let plot: (HTMLElement | SVGSVGElement) & Plot.Plot;
    let legend: HTMLElement | SVGSVGElement | undefined;
    switch (options.vistype) {
      case "lr_heatmap":
        plot = Plot.plot({
          width: 1000,
          height: 500,
          x: { label: "peak_lr_value", tickFormat: (v) => Math.exp(v), domain: [-7.2, 5.2] },
          y: { grid: true, domain: [0, 1] },
          color: {
            // legend: true,
            label: "average test accuracy within cell",
            scheme: "magma",
            type: "pow",
            exponent: 3,
            // ticks: [0.2, 0.5, 0.6, 0.7, 0.8, 0.9],
            tickFormat: (d: number) => {
              if (options.min_value === 0.9) {
                return `${Math.floor(d * 1000) / 10}%`;
              } else {
                return `${Math.floor(d * 100)}%`;
              }
            },
          },
          marks: [
            outerDot(
              values,
              innerBin({ fill: (d: Row[]) => color_by_average_test_accuracy(d, options) }, { x: (d: Row) => Math.log(d.peak_lr_value), y: "lr_decay", binWidth, thresholds })
            ),
          ],
        });
        legend = plot.legend("color", { width: 1000, height: 60 });
        break;
      case "nesterov_vs_max_accuracy":
        plot = Plot.plot({
          width: 1000,
          height: 500,
          x: { grid: true, label: "test accuracy: max(with nesterov, without nesterov)" },
          y: { grid: true },
          color: { tickFormat: (v) => Math.round(Math.exp(v)), label: "occurrences" },
          marks: [
            outerDot(
              values,
              innerBin({ fill: (d: Row[]) => Math.log(d.length), r: (d: Row[]) => Math.log(d.length) }, { x: "nmax", y: "diff_using_nesterov", binWidth, thresholds })
            ),
          ],
        });
        legend = plot.legend("color", { width: 1000, height: 60 });
        break;
      case "other":
        plot = Plot.plot({
          // height: 500,
          color: { legend: true },
          marks: [Plot.dot(values, { x: "nmax", y: "diff_using_nesterov", stroke: "diff_using_nesterov" })],
        });
        break;
      default:
        throw new Error(`unexpected visualization type: ${vistype.value}`);
    }
    // const plot = Plot.plot({
    //   // height: 500,
    //   color: { legend: true },
    //   // x: { label: "peak_lr_value", tickFormat: (v) => Math.exp(v), domain: [-7.2, 5.2] },
    //   // y: { grid: true, domain: [0, 1] },
    //   // marks: [Plot.dot(values, { x: "nmax", y: "diff_using_nesterov", stroke: "diff_using_nesterov" })],
    //   // marks: [
    //   //   Plot.barX(
    //   //     values,
    //   //     Plot.groupY({ x: "count" }, { y: "diff_bucketized", fill: "diff_using_nesterov", order: ["-0.5+", "-0.5 to -0.1", "-0.1 to +0.1", "+0.1 to +0.5", "+0.5+"] })
    //   //   ),
    //   // ],
    // });

    if (legend) {
      containerRef.current?.append(legend);
    }
    containerRef.current?.append(plot);
    return () => {
      legend?.remove();
      plot.remove();
    };
  }, [data, options]);

  return (
    <div>
      <div ref={containerRef} />
      <div>
        <select onChange={vistype.onChange} value={vistype.value}>
          <optgroup label="Visualization">
            <option value="lr_heatmap">Learning rate heatmap</option>
            <option value="nesterov_vs_max_accuracy">Nesterov momentum impact vs. max test accuracy</option>
            {/* <option value="other">Other</option> */}
          </optgroup>
        </select>
      </div>
      <div>
        <select onChange={binning.onChange} value={binning.value}>
          <optgroup label="Data binning">
            <option value="tiny">Tiny aggregation regions</option>
            <option value="small">Small aggregation regions</option>
            <option value="medium">Medium aggregation regions</option>
            <option value="large">Large aggregation regions</option>
            <option value="huge">Huge aggregation regions</option>
          </optgroup>
        </select>
        <label>
          <input type="checkbox" onChange={hex_aggregation.onChange} checked={hex_aggregation.value}></input>
          Use hexagonal aggregation regions
        </label>
      </div>
      <div>
        <select onChange={filter_nmax.onChange} value={filter_nmax.value}>
          <optgroup label="Data filtering">
            <option value="0.0">All data points</option>
            <option value="0.5">Results &ge; 50% accuracy</option>
            <option value="0.8">Results &ge; 80% accuracy</option>
            <option value="0.85">Results &ge; 85% accuracy</option>
            <option value="0.9">Results &ge; 90% accuracy</option>
          </optgroup>
        </select>
      </div>
      {vistype.value === "lr_heatmap" && (
        <select onChange={coloring_by.onChange} value={coloring_by.value}>
          <optgroup label="Color by test accuracy">
            <option value="best">Test accuracy (with or without Nesterov momentum, whichever is better)</option>
            <option value="only_nesterov">Test accuracy when using Nesterov momentum</option>
            <option value="never_nesterov">Test accuracy when not using Nesterov momentum</option>
            <option value="worst">Test accuracy (with or without Nesterov momentum, whichever is worse)</option>
          </optgroup>
          <optgroup label="Color by test accuracy difference">
            <option value="diff_nesterov">Difference in test accuracy by using Nesterov momentum</option>
          </optgroup>
        </select>
      )}
      {vistype.value === "nesterov_vs_max_accuracy" && (
        <select onChange={coloring_by.onChange} value={coloring_by.value === "diff_nesterov" ? "worst" : coloring_by.value}>
          <optgroup label="X axis">
            <option value="best">Test accuracy (with or without Nesterov momentum, whichever is better)</option>
            <option value="only_nesterov">Test accuracy when using Nesterov momentum</option>
            <option value="never_nesterov">Test accuracy when not using Nesterov momentum</option>
            <option value="worst">Test accuracy (with or without Nesterov momentum, whichever is worse)</option>
          </optgroup>
        </select>
      )}
      {/* <label>
        <input type="checkbox" onChange={divergent.onChange} checked={divergent.value}></input>
        Skip diverging regions
      </label> */}
      <div>
        Training procedure:
        <ul>
          <li>ResNet-20 trained on CIFAR-10 with standard data processing and augmentation (random crop, random left/right flip, global normalization)</li>
          <li>Using SGD with momentum and a cosine learning schedule with different hyperparameters, training for 100 epochs</li>
          <li>
            See <a href="https://github.com/houeland/tpu-ml/blob/main/cifar-10-optimization/extracted-sample/cifar10-resnet20.py">cifar10-resnet20.py</a> source code for exact
            details
          </li>
        </ul>
      </div>
    </div>
  );
}

{
  /* <option value="lr_heatmap">Learning rate heatmap</option>
x: peak_lr_value
y: lr_decay
color: test_accuracy

<option value="nesterov_vs_max_accuracy">Nesterov momentum impact vs. max test accuracy</option>
x: test_accuracy
y: nesterov_diff
color: count

<option value="other">Other</option> */
}
