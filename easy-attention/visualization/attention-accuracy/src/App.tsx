import { ChangeEvent, PropsWithChildren, useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import * as Plot from "@observablehq/plot";

type EvalExample = {
  eval_line: string;
  solution_softmax: number;
};

type Sample = {
  num_examples_trained_on: number;
  latest_training_loss: number;
  eval_examples: EvalExample[];
};

type Row = {
  num_examples_trained_on: number;
  eval_idx: number;
  eval_line: string;
  solution_softmax: number;
};

function smoothRows(input: Row[]) {
  const output: Row[] = [];
  const N = 15;
  for (let i = 0; i < input.length; i += 1) {
    let sum = 0;
    let ctr = 0;
    for (let j = Math.max(0, i - N); j <= Math.min(input.length - 1, i + N); j += 1) {
      const scaling = (16 - Math.abs(j - i)) / 16;
      sum += input[j].solution_softmax * scaling;
      ctr += 1 * scaling;
    }
    output.push({
      num_examples_trained_on: input[i].num_examples_trained_on,
      eval_idx: input[i].eval_idx,
      eval_line: input[i].eval_line,
      solution_softmax: sum / ctr,
    });
  }
  return output;
}

function processData(data: Sample[], smoothing: boolean, only250k: boolean) {
  const map = new Map<number, Row[]>();
  for (const d of data) {
    for (const [idx_, ee] of d.eval_examples.entries()) {
      const idx = idx_ + 1;
      const v = map.get(idx) ?? [];
      map.set(idx, v);
      if (!only250k || d.num_examples_trained_on < 250000) {
        v.push({
          num_examples_trained_on: d.num_examples_trained_on,
          eval_idx: idx,
          eval_line: ee.eval_line,
          solution_softmax: ee.solution_softmax,
        });
      }
    }
  }
  const values: Row[] = [];
  for (const [_k, v] of map) {
    if (smoothing) {
      values.push(...smoothRows(v));
    } else {
      values.push(...v);
    }
  }
  return values;
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

function Qq(props: PropsWithChildren<{}>) {
  return <span className="quotedtext">{props.children}</span>;
}

function Ee(props: PropsWithChildren<{}>) {
  return (
    <b>
      <i>{props.children}</i>
    </b>
  );
}

export function App() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<Sample[]>();
  const smoothing = useSelect("smoothed");
  const only250k = useCheckbox(true);

  useEffect(() => {
    async function load() {
      const response = await fetch("./training_log_solution_predictions.json");
      const data = (await response.json()) as Sample[];
      setData(data);
    }
    load();
  }, []);

  useEffect(() => {
    if (data === undefined) return;
    const values = processData(data, smoothing.value === "smoothed", only250k.value);
    const plot = Plot.plot({
      style: "overflow: visible;",
      y: { grid: true },
      marks: [
        Plot.lineY(values, { x: "num_examples_trained_on", y: "solution_softmax", stroke: "eval_line" }),
        Plot.tip(
          values,
          Plot.pointer({
            x: "num_examples_trained_on",
            y: "solution_softmax",
            title: (d) => `Eval #${d.eval_idx} sentence: ${d.eval_line}

num_examples_trained_on: ${d.num_examples_trained_on}
solution_softmax: ${d.solution_softmax.toFixed(5)}
`,
          })
        ),
      ],
    });
    containerRef.current?.append(plot);
    return () => plot.remove();
  }, [data, smoothing.value, only250k.value]);

  const evals = [
    { idx: 1, color: "red", prefix: 'rn biggest number in this list: 1 4 5 9", "solution": "', solution: "9" },
    { idx: 2, color: "green", prefix: 'rn smallest number in this list: 1 4 5 9", "solution": "', solution: "1" },
    { idx: 3, color: "blue", prefix: 'number in this list: 1 4 5 9", "solution": "', solution: "9 (though equally likely to be 1)" },
    { idx: 4, color: "cyan", prefix: 'rn biggest number in this list: 8 3 4 7", "solution": "', solution: "8" },
    { idx: 5, color: "yellow", prefix: 'rn smallest number in this list: 8 3 4 7", "solution": "', solution: "3" },
    { idx: 6, color: "orange", prefix: 'number in this list: 8 3 4 7", "solution": "', solution: "3 (though equally likely to be 8)" },
  ];
  return (
    <div>
      <div ref={containerRef} />
      <select onChange={smoothing.onChange}>
        <optgroup label="Dataset">
          <option selected value="smoothed">
            Single run (smoothed)
          </option>
          <option value="raw">Single run (raw values)</option>
        </optgroup>
      </select>
      <label>
        <input type="checkbox" onChange={only250k.onChange} checked={only250k.value}></input>
        Limit to initial 250k examples
      </label>
      <p>
        Training procedure:
        <ul>
          <li>Results from training a byte-level transformer language model on substrings of a dataset with questions about lists of single-digit numbers.</li>
          <li>
            There are random variations in how the questions are written, e.g. sometimes <Qq>Return biggest number: 1 2 3</Qq>, sometimes{" "}
            <Qq>Determine which is the maximum number among: 1 2 3</Qq>.
          </li>
          <li>For the results shown here, only questions asking for the smallest or largest number were used for training.</li>
          <li>
            Training example are 65-byte substrings of the full dataset, usually <Ee>not</Ee> starting at the beginning of a question.
          </li>
          <li>
            Training loss is measured equally for all bytes: each 65-byte substring results in 64 next-byte predictions, whose prediction loss are averaged. The solution to the
            biggest/smallest questions isn't "special" during training - it's just another byte.
          </li>
          <li>
            A training example substring to predict might not include the solution to any biggest/smallest question, or it might include a solution to predict but with insufficient
            preceding byte context for there to be a clear solution. (E.g. there could be 0 bytes from the question included in the training example!)
          </li>
          <li>
            The Adam optimizer is used for training, with learning rate=3e-4, beta1=0.9, beta2=0.99, epsilon=1e-08. Additionally updates are clipped with a maximum global norm of
            1.0.
          </li>
        </ul>
      </p>
      <p>
        Evaluation test cases:
        <ul>
          {evals.map((e) => {
            return (
              <li>
                Eval #{e.idx} ({e.color}): Predict the next character given input <Qq>{e.prefix}</Qq>, correct answer is {e.solution}.
              </li>
            );
          })}
        </ul>
        Eval #1 and Eval #2 are basically the same, and Eval #4 and Eval #5 are basically the same.<br></br>Eval #3 and #6 are markedly different though: they're both ambiguous and
        can't be "solved" above 50%, but Eval #6 requires understanding the list of numbers, and Eval #3 does not. (Similar to the difference between #4&#5 which requires
        understanding the numbers vs. #1&#2 which does not.)
      </p>
      <p>
        The language model's capabilities go through several stages during training:
        <ol>
          <li>
            Learn basic vocabulary, e.g. after <Qq>maximu</Qq> the next character is <Qq>m</Qq>, after <Qq>", "soluti</Qq> the next character is <Qq>o</Qq>, and that the solution
            bytes are always numbers. (This stage doesn't show up clearly in the results shown here, but it's fairly quick and happens first.)
          </li>
          <li>
            After around 40k examples, learn which task should be solved: finding the biggest number or finding the smallest number, and connecting that task to predicting the
            solution. At this point the model always predicts <Qq>1</Qq> or <Qq>9</Qq>, and starts getting good results on Eval #1 and Eval #2, and ~50% score on Eval #3. At the
            same time, the model does also learn that 1 and 9 are not always the right answer, plus some understanding that the numbers in the list are relevant, but doesn't yet
            have a clear understanding of the combination.
          </li>
          <li>
            After around 90k examples, the model learns how to combine the numbers in the list with the task. Eval #4 and Eval #5 are now well understood, though for some time with
            less certainty than Eval #1 and Eval #2. Eval #6 is understood at the same time, reaching ~50% score.
          </li>
          <li>
            After learning these tasks, the model is nearly-perfect, but with this current training setup it does <Ee>not</Ee> fully stabilize. The estimates for the ambiguous Eval
            #3 and Eval #6 cases are correctly averaging around 50% over time, but fluctuating wildly up and down around that average - presumably depending on which variant has
            been seen the most recently. And worse than that, there are regular "catastrophic" failures where individual predictions get really bad, e.g. suddenly a &lt;1%
            prediction for the correct solution to Eval #1 after 1.2M examples and Eval #5 after 1.5M examples, with regular dips below 90% even when the average is 99%+.
          </li>
        </ol>
      </p>
    </div>
  );
}
