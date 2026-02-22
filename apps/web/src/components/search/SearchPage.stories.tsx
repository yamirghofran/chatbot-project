import { useState, useEffect } from "react";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { SearchPage, type SearchPageProps } from "./SearchPage";
import { Navbar } from "@/components/navigation/Navbar";
import { mockBooks, mockUser } from "@/lib/mockData";
import type { Book } from "@/lib/types";

const meta = {
  title: "Search/SearchPage",
  component: SearchPage,
  parameters: {
    layout: "fullscreen",
    withContainer: false,
  },
  args: {
    query: "a slow burn romance set in rural Scotland",
    followUpSuggestions: [
      "Something shorter",
      "More recent",
      "Darker tone",
      "Pre-WWII setting",
      "Less romance, more landscape",
    ],
  },
} satisfies Meta<typeof SearchPage>;

export default meta;
type Story = StoryObj<typeof meta>;

const aiPicks = [mockBooks[0], mockBooks[2], mockBooks[9], mockBooks[7]];
const moreResults = [mockBooks[3], mockBooks[5], mockBooks[7]];

const aiNarrative =
  "You're describing something unhurried — a romance where the setting does as much emotional work as the characters. These tend to be quiet books with a strong sense of place, often literary rather than genre. Here are some that fit that mood:";

function SearchPageLayout(props: SearchPageProps) {
  return (
    <div className="min-h-dvh bg-background text-foreground">
      <Navbar user={mockUser} searchValue={props.query} />
      <main className="mx-auto max-w-5xl px-4 py-8">
        <SearchPage {...props} />
      </main>
    </div>
  );
}

// ─── Stories ────────────────────────────────────────────────────────────────

/**
 * Semantic query — no direct hit. AI section is dominant.
 * Keyword rows appear below as "More results".
 */
export const Complete: Story = {
  render: (args) => <SearchPageLayout {...args} />,
  args: {
    aiNarrative,
    aiBooks: aiPicks,
    keywordResults: moreResults,
  },
};

/**
 * Direct title/author hit at the top, then AI section, then more results.
 * The expected shape when the user searches something unambiguous.
 */
export const WithDirectHit: Story = {
  render: (args) => <SearchPageLayout {...args} />,
  args: {
    query: "The Remains of the Day",
    directHit: mockBooks[3],
    aiNarrative:
      "Fans of Kazuo Ishiguro often reach for books with the same quiet devastation — novels where what's left unsaid carries more weight than what is:",
    aiBooks: [mockBooks[5], mockBooks[0], mockBooks[7]],
    keywordResults: [mockBooks[2], mockBooks[9]],
    followUpSuggestions: [
      "More Japanese authors",
      "Something shorter",
      "Less restrained",
    ],
  },
};

/**
 * AI is still generating. Direct hit is already visible at top.
 */
export const Loading: Story = {
  render: (args) => <SearchPageLayout {...args} />,
  args: {
    directHit: mockBooks[3],
    isAiLoading: true,
  },
};

/**
 * A vague semantic query — no direct hit, no keyword rows.
 * AI section is the entire result.
 */
export const SemanticOnly: Story = {
  render: (args) => <SearchPageLayout {...args} />,
  args: {
    query: "books that made me cry on a plane",
    aiNarrative:
      "These books have a habit of arriving at the emotional sucker punch when you least expect it — perfect for altitude-induced vulnerability:",
    aiBooks: [mockBooks[7], mockBooks[3], mockBooks[10], mockBooks[5]],
    keywordResults: [],
    followUpSuggestions: [
      "Shorter",
      "Something funny instead",
      "Nonfiction version",
    ],
  },
};

/**
 * Exact author search — direct hit is the author's most popular book,
 * AI adds "fans also like", keyword rows list their other works.
 */
export const AuthorSearch: Story = {
  render: (args) => <SearchPageLayout {...args} />,
  args: {
    query: "Haruki Murakami",
    directHit: mockBooks[2],
    aiNarrative:
      "Fans of Murakami often reach for books with the same dreamlike quality and emotional restraint — stories where the surface calm conceals something stranger underneath:",
    aiBooks: [mockBooks[3], mockBooks[9], mockBooks[0]],
    keywordResults: [],
    followUpSuggestions: ["More surreal", "Something shorter", "More grounded"],
  },
};

/**
 * Simulates streaming — keyword direct hit shows immediately,
 * AI narrative types in, book results stagger in.
 */
export const Streaming: Story = {
  render: (args) => {
    const [displayedText, setDisplayedText] = useState("");
    const [displayedBooks, setDisplayedBooks] = useState<Book[]>([]);
    const [followUp, setFollowUp] = useState("");

    useEffect(() => {
      setDisplayedText("");
      setDisplayedBooks([]);
      let charIndex = 0;

      const textTimer = setInterval(() => {
        charIndex++;
        setDisplayedText(aiNarrative.slice(0, charIndex));
        if (charIndex >= aiNarrative.length) {
          clearInterval(textTimer);
          aiPicks.forEach((book, i) => {
            setTimeout(
              () => {
                setDisplayedBooks((prev) => [...prev, book]);
              },
              (i + 1) * 200,
            );
          });
        }
      }, 18);

      return () => clearInterval(textTimer);
    }, []);

    return (
      <SearchPageLayout
        {...args}
        directHit={mockBooks[3]}
        keywordResults={moreResults}
        isAiLoading={displayedText.length === 0}
        aiNarrative={displayedText || undefined}
        aiBooks={displayedBooks}
        followUpValue={followUp}
        onFollowUpChange={setFollowUp}
        onFollowUpSubmit={(v) => alert(`Follow-up: "${v}"`)}
      />
    );
  },
};

/** No results at all. */
export const NoResults: Story = {
  render: (args) => <SearchPageLayout {...args} />,
  args: {
    query: "xyzzy frobozz zork",
    directHit: undefined,
    keywordResults: [],
    isAiLoading: false,
    aiNarrative: undefined,
    aiBooks: [],
    followUpSuggestions: [],
  },
};
