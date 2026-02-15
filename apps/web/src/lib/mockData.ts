import type { Book, User, List, ActivityItem } from "./types";

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
  {
    id: "12",
    title: "The Vegetarian",
    author: "Han Kang",
    coverUrl: "https://covers.openlibrary.org/b/isbn/9781101906101-L.jpg",
    description:
      "After a disturbing dream, a quiet South Korean woman refuses to eat meat. Her seemingly minor act of rebellion spirals outward, exposing the violence that underpins her family and society. Told in three interconnected novellas.",
    tags: ["Fiction", "Literary"],
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
    name: "Favourites of 2025",
    owner: mockUser,
    books: mockBooks.slice(0, 5),
  },
  {
    id: "l2",
    name: "To Read This Summer",
    owner: mockUser,
    books: mockBooks.slice(3, 9),
  },
  {
    id: "l3",
    name: "Best Nonfiction",
    owner: mockUser,
    books: [mockBooks[1], mockBooks[4], mockBooks[6], mockBooks[8], mockBooks[10]],
  },
];

export const mockActivity: ActivityItem[] = [
  {
    id: "a1",
    user: mockFriends[0],
    type: "rating",
    book: mockBooks[1],
    rating: 4.5,
    timestamp: "2h ago",
  },
  {
    id: "a2",
    user: mockFriends[1],
    type: "favourite",
    book: mockBooks[11],
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
    type: "favourite",
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

export const mockStaffPicks: Book[] = [
  mockBooks[0],
  mockBooks[2],
  mockBooks[5],
  mockBooks[7],
  mockBooks[9],
  mockBooks[11],
];
