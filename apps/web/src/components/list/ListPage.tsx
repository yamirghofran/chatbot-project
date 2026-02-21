import { useState } from "react";
import { GripVertical, X, Pencil, Check, List as ListIcon, LayoutGrid, Trash2 } from "lucide-react";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import type { Book, List } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Toggle } from "@/components/ui/toggle";
import { ListBookRow } from "@/components/list/ListBookRow";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/ui/avatar";

type ViewMode = "list" | "grid";

export type ListPageProps = {
  list: List;
  isOwner?: boolean;
  defaultViewMode?: ViewMode;
  onReorder?: (bookIds: string[]) => void;
  onRemoveBook?: (bookId: string) => void;
  onAddBook?: (book: Book) => void;
  onUpdateName?: (name: string) => void;
  onUpdateDescription?: (description: string) => void;
  onDeleteList?: () => void;
};

function SortableBookRow({
  book,
  isEditing,
  onRemove,
}: {
  book: Book;
  isEditing: boolean;
  onRemove?: () => void;
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: book.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  return (
    <div ref={setNodeRef} style={style} className="flex items-center">
      {isEditing && (
        <button
          type="button"
          className="shrink-0 p-1 text-muted-foreground cursor-grab active:cursor-grabbing"
          aria-label="Drag to reorder"
          {...attributes}
          {...listeners}
        >
          <GripVertical className="size-4" />
        </button>
      )}
      <div className="flex-1 min-w-0">
        <ListBookRow book={book} compact />
      </div>
      {isEditing && (
        <button
          type="button"
          className="shrink-0 p-2 text-muted-foreground hover:text-foreground transition-colors"
          aria-label={`Remove ${book.title}`}
          onClick={onRemove}
        >
          <X className="size-4" />
        </button>
      )}
    </div>
  );
}

export function ListPage({
  list,
  isOwner = false,
  defaultViewMode = "list",
  onReorder,
  onRemoveBook,
  onUpdateName,
  onUpdateDescription,
  onDeleteList,
}: ListPageProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>(defaultViewMode);
  const [editName, setEditName] = useState(list.name);
  const [editDescription, setEditDescription] = useState(
    list.description ?? "",
  );
  const [descriptionExpanded, setDescriptionExpanded] = useState(false);

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  );

  function handleToggleEdit() {
    if (isEditing) {
      if (editName !== list.name) onUpdateName?.(editName);
      if (editDescription !== (list.description ?? ""))
        onUpdateDescription?.(editDescription);
      setIsEditing(false);
    } else {
      setEditName(list.name);
      setEditDescription(list.description ?? "");
      setIsEditing(true);
    }
  }

  function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event;
    if (over && active.id !== over.id) {
      const oldIndex = list.books.findIndex((b) => b.id === active.id);
      const newIndex = list.books.findIndex((b) => b.id === over.id);
      const reordered = arrayMove(list.books, oldIndex, newIndex);
      onReorder?.(reordered.map((b) => b.id));
    }
  }

  const ownerInitials = list.owner.displayName
    .split(" ")
    .map((n) => n[0])
    .join("")
    .slice(0, 2);

  return (
    <div>
      {/* Header */}
      <div className="flex items-start justify-between gap-4 mb-1">
        <div className="min-w-0 flex-1">
          {isEditing ? (
            <input
              type="text"
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              className="font-heading text-xl font-semibold text-foreground bg-transparent border-b border-border outline-none w-full"
            />
          ) : (
            <h1 className="font-heading text-xl font-semibold text-foreground">
              {list.name}
            </h1>
          )}
        </div>
        {isOwner && (
          <Button variant="ghost" size="sm" onClick={handleToggleEdit}>
            {isEditing ? (
              <>
                <Check className="size-4" />
                Done
              </>
            ) : (
              <>
                <Pencil className="size-4" />
                Edit
              </>
            )}
          </Button>
        )}
      </div>

      {/* Owner + count */}
      <div className="flex items-center gap-2 mb-4">
        <Avatar size="sm">
          {list.owner.avatarUrl && (
            <AvatarImage src={list.owner.avatarUrl} alt={list.owner.displayName} />
          )}
          <AvatarFallback>{ownerInitials}</AvatarFallback>
        </Avatar>
        <span className="text-sm text-muted-foreground">
          {list.owner.displayName}
        </span>
        <span className="text-sm text-muted-foreground">&middot;</span>
        <span className="text-sm text-muted-foreground">
          {list.books.length} {list.books.length === 1 ? "book" : "books"}
        </span>
      </div>

      {/* Description */}
      {isEditing ? (
        <textarea
          value={editDescription}
          onChange={(e) => setEditDescription(e.target.value)}
          placeholder="Tell the story behind this list…"
          rows={3}
          className="w-full text-sm text-foreground bg-transparent border border-border rounded-md p-2 mb-4 outline-none resize-none focus:ring-1 focus:ring-ring"
        />
      ) : (
        list.description && (
          <div className="mb-4">
            <p
              className={`text-sm text-muted-foreground ${!descriptionExpanded ? "line-clamp-3" : ""}`}
            >
              {list.description}
            </p>
            {!descriptionExpanded && list.description.length > 200 && (
              <button
                type="button"
                className="text-sm text-foreground font-medium mt-1"
                onClick={() => setDescriptionExpanded(true)}
              >
                Show more
              </button>
            )}
          </div>
        )
      )}

      {/* View mode toggle */}
      {!isEditing && list.books.length > 0 && (
        <div className="flex items-center gap-1 mb-4">
          <Toggle
            variant="outline"
            size="sm"
            pressed={viewMode === "list"}
            onPressedChange={() => setViewMode("list")}
            aria-label="List view"
          >
            <ListIcon className="size-4" />
          </Toggle>
          <Toggle
            variant="outline"
            size="sm"
            pressed={viewMode === "grid"}
            onPressedChange={() => setViewMode("grid")}
            aria-label="Grid view"
          >
            <LayoutGrid className="size-4" />
          </Toggle>
        </div>
      )}

      {/* Book list */}
      {list.books.length === 0 ? (
        <div className="flex flex-col items-center rounded-xl bg-card p-8">
          <img
            src="/brand/cartoon-reading2.jpg"
            alt=""
            className="w-32"
          />
          <p className="text-sm text-muted-foreground mt-4">
            This list is empty. Add some books to get started.
          </p>
        </div>
      ) : isEditing ? (
        <DndContext
          sensors={sensors}
          collisionDetection={closestCenter}
          onDragEnd={handleDragEnd}
        >
          <SortableContext
            items={list.books.map((b) => b.id)}
            strategy={verticalListSortingStrategy}
          >
            {list.books.map((book, i) => (
              <div key={book.id}>
                {i > 0 && <Separator />}
                <SortableBookRow
                  book={book}
                  isEditing
                  onRemove={() => onRemoveBook?.(book.id)}
                />
              </div>
            ))}
          </SortableContext>
        </DndContext>
      ) : viewMode === "grid" ? (
        <div className="grid grid-cols-3 gap-4">
          {list.books.map((book) => (
            <div key={book.id} className="min-w-0">
              <img
                src={book.coverUrl ?? "/brand/book-placeholder.png"}
                alt={`Cover of ${book.title}`}
                className="w-full aspect-[2/3] rounded-md object-cover"
              />
              <p className="text-sm font-medium text-foreground mt-2 truncate">
                {book.title}
              </p>
              <p className="text-xs text-muted-foreground truncate">
                {book.author}
              </p>
            </div>
          ))}
        </div>
      ) : (
        <div>
          {list.books.map((book, i) => (
              <div key={book.id}>
                {i > 0 && <Separator />}
                <ListBookRow book={book} />
              </div>
            ))}
          </div>
      )}

      {/* Add book stub (edit mode only) */}
      {isEditing && (
        <div className="mt-4">
          <Separator className="mb-4" />
          <div className="flex items-center gap-2 rounded-md border border-dashed border-border p-3 text-sm text-muted-foreground">
            Add a book…
          </div>
          <div className="mt-3 flex justify-end">
            <Button
              type="button"
              variant="destructive"
              size="sm"
              onClick={onDeleteList}
            >
              <Trash2 className="size-4" />
              Delete list
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
