import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { mockFetch, stats } from "../test/fixtures";
import { StatsPanel } from "./StatsPanel";

afterEach(() => vi.unstubAllGlobals());

describe("StatsPanel", () => {
  it("shows the live numbers", () => {
    render(<StatsPanel stats={stats} error={null} onCleared={vi.fn()} />);
    expect(screen.getByText("75.0%")).toBeInTheDocument();
    expect(screen.getByText("3 of 4 queries")).toBeInTheDocument();
    expect(screen.getByText("13.7 ms")).toBeInTheDocument();
    expect(screen.getByText("p95 70.1 ms")).toBeInTheDocument();
    expect(screen.getByText("Similarity threshold 0.70")).toBeInTheDocument();
  });

  it("shows dashes instead of zeros before any traffic", () => {
    render(
      <StatsPanel
        stats={{ ...stats, total_queries: 0, cache_hits: 0, cache_misses: 0 }}
        error={null}
        onCleared={vi.fn()}
      />,
    );
    expect(screen.getByText("no hits yet")).toBeInTheDocument();
    expect(screen.getByText("no misses yet")).toBeInTheDocument();
  });

  it("clears the cache with the admin token", async () => {
    const fetchMock = mockFetch({ body: { removed: 5 } });
    const onCleared = vi.fn();
    render(<StatsPanel stats={stats} error={null} onCleared={onCleared} />);

    await userEvent.click(screen.getByText("Clear the cache (admin)"));
    await userEvent.type(screen.getByLabelText("Admin token"), "s3cret");
    await userEvent.click(screen.getByRole("button", { name: "Clear" }));

    expect(await screen.findByText("Cleared 5 cached answers.")).toBeInTheDocument();
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/cache");
    expect(init.method).toBe("DELETE");
    expect(init.headers["X-Admin-Token"]).toBe("s3cret");
    expect(onCleared).toHaveBeenCalled();
  });
});
