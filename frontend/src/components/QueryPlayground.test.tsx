import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { hit, mockFetch, vetoed } from "../test/fixtures";
import { QueryPlayground } from "./QueryPlayground";

afterEach(() => vi.unstubAllGlobals());

describe("QueryPlayground", () => {
  it("sends the question and shows a cache hit", async () => {
    const fetchMock = mockFetch({ body: hit });
    const onResult = vi.fn();
    render(<QueryPlayground threshold={0.7} onResult={onResult} />);

    await userEvent.type(screen.getByLabelText("Question"), "forgot password help");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText("HIT")).toBeInTheDocument();
    expect(screen.getByText("“How do I reset my password?”")).toBeInTheDocument();
    expect(screen.getByText("0.88")).toBeInTheDocument();
    expect(screen.getByText("13.7 ms")).toBeInTheDocument();

    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/query");
    expect(JSON.parse(init.body)).toEqual({ query: "forgot password help", use_cache: true });
    expect(onResult).toHaveBeenCalledWith("forgot password help", hit);
  });

  it("explains a guard veto", async () => {
    mockFetch({ body: vetoed });
    render(<QueryPlayground threshold={0.7} onResult={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: "How do I NOT reset my password?" }));

    expect(await screen.findByText("MISS")).toBeInTheDocument();
    expect(screen.getByText(/Blocked by guard: Negation differs/)).toBeInTheDocument();
    expect(screen.getByText("account_and_auth.md#2")).toBeInTheDocument();
  });

  it("can bypass the cache", async () => {
    const fetchMock = mockFetch({ body: { ...hit, cache_status: "bypass", cache_hit: false } });
    render(<QueryPlayground threshold={0.7} onResult={vi.fn()} />);
    await userEvent.click(screen.getByRole("checkbox", { name: "Use cache" }));
    await userEvent.click(screen.getByRole("button", { name: "Do you offer refunds?" }));

    expect(await screen.findByText("BYPASS")).toBeInTheDocument();
    expect(JSON.parse(fetchMock.mock.calls[0]![1].body).use_cache).toBe(false);
  });

  it("shows the API's error message", async () => {
    mockFetch({ status: 502, body: { detail: "LLM provider timed out", request_id: "x" } });
    render(<QueryPlayground threshold={0.7} onResult={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: "Do you offer refunds?" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("LLM provider timed out");
  });
});
