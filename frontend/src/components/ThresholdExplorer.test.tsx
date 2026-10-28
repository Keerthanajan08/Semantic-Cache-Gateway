import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { evalResults } from "../test/fixtures";
import { ThresholdExplorer } from "./ThresholdExplorer";

function readout(label: string) {
  return within(screen.getByText(label, { selector: ".readout .label" }).parentElement!).getByText(
    /%$/,
  );
}

describe("ThresholdExplorer", () => {
  it("starts at the evaluated default for the best model", () => {
    render(<ThresholdExplorer data={evalResults} liveThreshold={0.7} liveModel={null} />);
    expect(screen.getByRole("combobox")).toHaveValue("all-mpnet-base-v2");
    expect(readout("From cache")).toHaveTextContent("63.2%");
    expect(readout("Wrong answers")).toHaveTextContent("0.9%");
  });

  it("updates the readout when the slider moves", () => {
    render(<ThresholdExplorer data={evalResults} liveThreshold={0.7} liveModel={null} />);
    fireEvent.change(screen.getByRole("slider"), { target: { value: "0.72" } });
    expect(readout("From cache")).toHaveTextContent("59.4%");
    expect(readout("Wrong answers")).toHaveTextContent("0.0%");
    expect(screen.getByRole("button", { name: "Reset to default" })).toBeInTheDocument();
  });

  it("shows the unguarded numbers when guards are switched off", async () => {
    render(<ThresholdExplorer data={evalResults} liveThreshold={0.7} liveModel={null} />);
    await userEvent.click(screen.getByRole("checkbox", { name: /safety guards on/i }));
    expect(readout("Wrong answers")).toHaveTextContent("2.8%");
  });

  it("switches model and jumps to that model's default", async () => {
    render(<ThresholdExplorer data={evalResults} liveThreshold={0.7} liveModel={null} />);
    await userEvent.selectOptions(screen.getByRole("combobox"), "all-MiniLM-L6-v2");
    expect(screen.getByRole("slider")).toHaveValue("0.72");
    expect(readout("From cache")).toHaveTextContent("50.0%");
  });

  it("opens on the model the running gateway uses", () => {
    render(
      <ThresholdExplorer data={evalResults} liveThreshold={0.72} liveModel="all-MiniLM-L6-v2" />,
    );
    expect(screen.getByRole("combobox")).toHaveValue("all-MiniLM-L6-v2");
    expect(screen.getByText(/running gateway uses 0.72/)).toBeInTheDocument();
  });

  it("offers the data as a table", () => {
    render(<ThresholdExplorer data={evalResults} liveThreshold={0.7} liveModel={null} />);
    expect(screen.getByText("0.70 (default)")).toBeInTheDocument();
  });
});
