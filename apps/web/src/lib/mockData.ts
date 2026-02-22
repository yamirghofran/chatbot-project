import type {
  Book,
  User,
  List,
  ActivityItem,
  RatedBook,
  BookStats,
  Review,
} from "./types";

export const mockBooks: Book[] = [
  {
    id: "1",
    title: "The Master and Margarita",
    author: "Mikhail Bulgakov",
    coverUrl: "https://covers.openlibrary.org/b/isbn/9780143108276-L.jpg",
    description:
      "The Devil arrives in Soviet Moscow accompanied by a retinue that includes a giant cat, a fanged assassin, and a naked witch. As he wreaks havoc on the city's literary establishment, a parallel narrative follows Pontius Pilate's fateful encounter with a wandering philosopher.",
    tags: ["Fiction", "Classic", "Satire", "Russian"],
  },
  {
    id: "2",
    title: "Sapiens",
    author: "Yuval Noah Harari",
    coverUrl: "https://covers.openlibrary.org/b/isbn/9780062316097-L.jpg",
    description:
      "A sweeping history of humankind from the emergence of Homo sapiens in Africa to the present. Harari examines how biology, mythology, and economics have shaped human societies, asking what it means to be human and where our species is heading.",
    tags: ["Nonfiction", "History", "Anthropology", "Science"],
  },
  {
    id: "3",
    title: "Kafka on the Shore",
    author: "Haruki Murakami",
    coverUrl: "https://covers.openlibrary.org/b/isbn/9781400079278-L.jpg",
    description:
      "A fifteen-year-old boy runs away from home to escape an oedipal prophecy, while an elderly man who can talk to cats sets out on a journey of his own. Their stories intertwine in a dreamlike narrative blurring the boundaries between the real and the fantastic.",
    tags: ["Fiction", "Surrealist", "Japanese", "Coming-of-age"],
  },
  {
    id: "4",
    title: "The Remains of the Day",
    author: "Kazuo Ishiguro",
    coverUrl: "https://covers.openlibrary.org/b/isbn/9780679731726-L.jpg",
    description:
      "An ageing English butler takes a motoring trip through the West Country, reflecting on his decades of service at a great house. Beneath his impeccable composure lies a lifetime of suppressed emotion and missed opportunities.",
    tags: ["Fiction", "Literary"],
  },
  {
    id: "5",
    title: "Thinking, Fast and Slow",
    author: "Daniel Kahneman",
    coverUrl: "https://covers.openlibrary.org/b/isbn/9780374533557-L.jpg",
    description:
      "Nobel laureate Daniel Kahneman reveals the two systems that drive the way we think—fast, intuitive, and emotional versus slow, deliberate, and logical—and shows how cognitive biases shape our judgments and decisions.",
    tags: ["Nonfiction", "Psychology"],
  },
  {
    id: "6",
    title: "Stoner",
    author: "John Williams",
    coverUrl: "https://covers.openlibrary.org/b/isbn/9781590171998-L.jpg",
    description:
      "William Stoner, the son of poor Missouri farmers, discovers a love of literature at university and spends his life as an English professor. A quiet, devastating portrait of an unremarkable life lived with hidden passion and dignity.",
    tags: ["Fiction", "Literary"],
  },
  {
    id: "7",
    title: "The Periodic Table",
    author: "Primo Levi",
    coverUrl: "https://covers.openlibrary.org/b/isbn/9780805210415-L.jpg",
    description:
      "Each chapter named after a chemical element, Levi weaves together autobiography, war memoir, and meditations on science. From his childhood in Turin to Auschwitz and beyond, the elements become metaphors for the human experience.",
    tags: ["Nonfiction", "Memoir"],
  },
  {
    id: "8",
    title: "Beloved",
    author: "Toni Morrison",
    coverUrl: "https://covers.openlibrary.org/b/isbn/9781400033416-L.jpg",
    description:
      "Set in post-Civil War Ohio, a former slave is haunted by the ghost of her dead daughter. Morrison explores the physical, emotional, and spiritual devastation wrought by slavery in prose that is both lyrical and unflinching.",
    tags: ["Fiction", "Classic", "Gothic", "American"],
  },
  {
    id: "9",
    title: "The Gene",
    author: "Siddhartha Mukherjee",
    coverUrl: "https://covers.openlibrary.org/b/isbn/9781476733524-L.jpg",
    description:
      "A history of the gene from Mendel's garden peas to CRISPR, intertwined with Mukherjee's own family story of mental illness. A gripping account of how genetics has transformed our understanding of identity, fate, and disease.",
    tags: ["Nonfiction", "Science"],
  },
  {
    id: "10",
    title: "Pedro Páramo",
    author: "Juan Rulfo",
    coverUrl: "https://covers.openlibrary.org/b/isbn/9780802133908-L.jpg",
    description:
      "A young man travels to the village of Comala in search of his father, only to find a ghost town populated by the murmuring dead. In barely 120 pages, Rulfo created a foundational work of Latin American literature.",
    tags: ["Fiction", "Magic Realism", "Mexican", "Novella"],
  },
  {
    id: "11",
    title: "When Breath Becomes Air",
    author: "Paul Kalanithi",
    coverUrl: "https://covers.openlibrary.org/b/isbn/9780812988406-L.jpg",
    description:
      "A neurosurgeon diagnosed with terminal lung cancer at thirty-six confronts the question of what makes life meaningful. Written in the final months of his life, it is a meditation on mortality, medicine, and literature.",
    tags: ["Nonfiction", "Memoir"],
  },
];

export const mockUser: User = {
  id: "u1",
  handle: "mporteous",
  displayName: "Matt Porteous",
  avatarUrl: undefined,
};

export const mockFriends: User[] = [
  {
    id: "u2",
    handle: "alexchen",
    displayName: "Alex Chen",
    avatarUrl: undefined,
  },
  {
    id: "u3",
    handle: "sarahkim",
    displayName: "Sarah Kim",
    avatarUrl: undefined,
  },
  {
    id: "u4",
    handle: "jamieross",
    displayName: "Jamie Ross",
    avatarUrl: undefined,
  },
];

export const mockLists: List[] = [
  {
    id: "l1",
    name: "Shelled in 2025",
    description:
      "The books that completely took over my year. Each one left a mark.",
    owner: mockUser,
    books: mockBooks.slice(0, 5),
  },
  {
    id: "l2",
    name: "To Read This Summer",
    description: "My ambitious summer reading pile. Let's see how far I get.",
    owner: mockUser,
    books: mockBooks.slice(3, 9),
  },
  {
    id: "l3",
    name: "Best Nonfiction",
    owner: mockUser,
    books: [
      mockBooks[1],
      mockBooks[4],
      mockBooks[6],
      mockBooks[8],
      mockBooks[10],
    ],
  },
];

export const mockManyLists: List[] = [
  ...mockLists,
  {
    id: "l4",
    name: "Short Books I Loved",
    owner: mockUser,
    books: mockBooks.slice(0, 3),
  },
  {
    id: "l5",
    name: "Long Winter Reads",
    owner: mockUser,
    books: mockBooks.slice(2, 8),
  },
  {
    id: "l6",
    name: "Philosophy & Ideas",
    owner: mockUser,
    books: [mockBooks[0], mockBooks[4], mockBooks[5]],
  },
  {
    id: "l7",
    name: "Books I Recommend First",
    owner: mockUser,
    books: mockBooks.slice(6, 11),
  },
  {
    id: "l8",
    name: "Read Again Someday",
    owner: mockUser,
    books: [mockBooks[2], mockBooks[3], mockBooks[8]],
  },
  {
    id: "l9",
    name: "Brilliant Nonfiction",
    owner: mockUser,
    books: [mockBooks[1], mockBooks[4], mockBooks[8], mockBooks[10]],
  },
  {
    id: "l10",
    name: "Quiet Literary Novels",
    owner: mockUser,
    books: [mockBooks[3], mockBooks[5], mockBooks[7]],
  },
  {
    id: "l11",
    name: "Books to Discuss",
    owner: mockUser,
    books: mockBooks.slice(0, 6),
  },
  {
    id: "l12",
    name: "All-time Comfort Reads",
    owner: mockUser,
    books: [mockBooks[0], mockBooks[2], mockBooks[9]],
  },
];

export const mockActivity: ActivityItem[] = [
  {
    id: "a1",
    user: mockFriends[0],
    type: "rating",
    book: mockBooks[1],
    rating: 5,
    timestamp: "2h ago",
  },
  {
    id: "a2",
    user: mockFriends[1],
    type: "shell_add",
    book: mockBooks[6],
    timestamp: "5h ago",
  },
  {
    id: "a3",
    user: mockFriends[2],
    type: "list_add",
    book: mockBooks[9],
    listName: "Summer Reads",
    timestamp: "1d ago",
  },
  {
    id: "a4",
    user: mockFriends[0],
    type: "shell_add",
    book: mockBooks[3],
    timestamp: "1d ago",
  },
  {
    id: "a5",
    user: mockFriends[1],
    type: "rating",
    book: mockBooks[6],
    rating: 5,
    timestamp: "2d ago",
  },
];

export const mockRatedBooks: RatedBook[] = [
  { book: mockBooks[0], rating: 5, ratedAt: "2025-12-01" },
  { book: mockBooks[1], rating: 4, ratedAt: "2025-11-28" },
  { book: mockBooks[2], rating: 5, ratedAt: "2025-11-20" },
  { book: mockBooks[3], rating: 5, ratedAt: "2025-11-15" },
  { book: mockBooks[4], rating: 3, ratedAt: "2025-11-10" },
  { book: mockBooks[5], rating: 5, ratedAt: "2025-10-30" },
  { book: mockBooks[6], rating: 4, ratedAt: "2025-10-22" },
  { book: mockBooks[7], rating: 5, ratedAt: "2025-10-15" },
  { book: mockBooks[8], rating: 3, ratedAt: "2025-10-08" },
  { book: mockBooks[9], rating: 4, ratedAt: "2025-09-25" },
  { book: mockBooks[10], rating: 2, ratedAt: "2025-09-18" },
  { book: mockBooks[10], rating: 4, ratedAt: "2025-09-10" },
];

export const mockFavorites: Book[] = [mockBooks[0], mockBooks[2], mockBooks[5]];

export const mockUserActivity: ActivityItem[] = [
  {
    id: "ua1",
    user: mockUser,
    type: "rating",
    book: mockBooks[0],
    rating: 5,
    timestamp: "1d ago",
  },
  {
    id: "ua2",
    user: mockUser,
    type: "shell_add",
    book: mockBooks[2],
    timestamp: "3d ago",
  },
  {
    id: "ua3",
    user: mockUser,
    type: "list_add",
    book: mockBooks[5],
    listName: "Shelled in 2025",
    timestamp: "5d ago",
  },
  {
    id: "ua4",
    user: mockUser,
    type: "rating",
    book: mockBooks[7],
    rating: 5,
    timestamp: "1w ago",
  },
  {
    id: "ua5",
    user: mockUser,
    type: "shell_add",
    book: mockBooks[9],
    timestamp: "2w ago",
  },
];

export const mockStaffPicks: Book[] = [
  mockBooks[0],
  mockBooks[2],
  mockBooks[5],
  mockBooks[7],
  mockBooks[9],
  mockBooks[10],
];

export const mockBookStats: BookStats = {
  averageRating: 4.3,
  ratingCount: 128,
  commentCount: 21,
  shellCount: 47,
};

export const mockBookActivity: ActivityItem[] = [
  {
    id: "ba1",
    user: mockFriends[0],
    type: "rating",
    book: mockBooks[0],
    rating: 5,
    timestamp: "3h ago",
  },
  {
    id: "ba2",
    user: mockFriends[1],
    type: "shell_add",
    book: mockBooks[0],
    timestamp: "1d ago",
  },
  {
    id: "ba3",
    user: mockFriends[2],
    type: "list_add",
    book: mockBooks[0],
    listName: "All-time Favourites",
    timestamp: "3d ago",
  },
];

export const mockReviews: Review[] = [
  {
    id: "r1",
    user: mockFriends[0],
    text: "An extraordinary novel that blends satire, fantasy, and philosophy into something completely unique. The scenes in Moscow are laugh-out-loud funny, while the Pilate chapters carry real weight. Bulgakov was decades ahead of his time.",
    likes: 12,
    isLikedByMe: true,
    timestamp: "2d ago",
    replies: [
      {
        id: "rp1",
        user: mockFriends[1],
        text: "Completely agree about the Pilate chapters. They hit differently on a second read.",
        likes: 3,
        timestamp: "1d ago",
      },
    ],
  },
  {
    id: "r2",
    user: mockFriends[2],
    text: "Started slow for me but by Part 2 I couldn't put it down. The ending is genuinely moving. One of those books that stays with you.",
    likes: 5,
    timestamp: "1w ago",
  },
  {
    id: "r3",
    user: {
      id: "u5",
      handle: "elliot",
      displayName: "Elliot Vance",
    },
    text: "I think this is one of the greatest novels of the 20th century. The way Bulgakov weaves multiple timelines and genres together is masterful. Woland is one of literature's best characters.",
    likes: 8,
    timestamp: "2w ago",
    replies: [
      {
        id: "rp2",
        user: mockFriends[0],
        text: "Woland steals every scene he's in. The ball chapter is unforgettable.",
        likes: 2,
        isLikedByMe: true,
        timestamp: "2w ago",
      },
      {
        id: "rp3",
        user: mockFriends[2],
        text: "Have you read the Pevear and Volokhonsky translation? Curious how it compares.",
        likes: 1,
        timestamp: "1w ago",
      },
    ],
  },
];
