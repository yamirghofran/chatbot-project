import { useEffect, useRef, useState, type ReactElement } from "react";
import { Plus } from "lucide-react";
import { cn } from "@/lib/utils";
import { SearchBar } from "@/components/search/SearchBar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export type AddToListMenuOption = {
  id: string;
  name: string;
  bookCount?: number;
};

export type AddToListMenuProps = {
  lists: AddToListMenuOption[];
  selectedListIds?: string[];
  onToggleList?: (listId: string, nextSelected: boolean) => void;
  onCreateList?: (name: string) => void;
  className?: string;
  align?: "start" | "end";
  trigger: ReactElement<{
    onClick?: () => void;
    "aria-haspopup"?: "menu";
    "aria-expanded"?: boolean;
  }>;
};

export function AddToListMenu({
  lists,
  selectedListIds,
  onToggleList,
  onCreateList,
  className,
  align = "start",
  trigger,
}: AddToListMenuProps) {
  const MAX_VISIBLE_LISTS = 4.5;
  const [open, setOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [newListName, setNewListName] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [internalSelected, setInternalSelected] = useState<Set<string>>(
    () => new Set(selectedListIds ?? []),
  );
  const inputRef = useRef<HTMLInputElement>(null);
  const isControlled = selectedListIds !== undefined;
  const selected = isControlled ? new Set(selectedListIds) : internalSelected;

  useEffect(() => {
    if (open && isCreating) {
      inputRef.current?.focus();
    }
  }, [open, isCreating]);

  function toggleList(listId: string) {
    const next = new Set(selected);
    const currentlySelected = next.has(listId);
    const nextSelected = !currentlySelected;
    if (nextSelected) {
      next.add(listId);
    } else {
      next.delete(listId);
    }
    if (!isControlled) {
      setInternalSelected(next);
    }
    onToggleList?.(listId, nextSelected);
  }

  function createList() {
    const name = newListName.trim();
    if (!name) return;
    onCreateList?.(name);
    setNewListName("");
    setIsCreating(false);
  }

  const normalizedQuery = searchQuery.trim().toLowerCase();
  const filteredLists = normalizedQuery
    ? lists.filter((list) => list.name.toLowerCase().includes(normalizedQuery))
    : lists;
  const hasSingleList = lists.length === 1;
  const isListAreaScrollable = filteredLists.length > MAX_VISIBLE_LISTS;

  return (
    <DropdownMenu
      open={open}
      onOpenChange={(nextOpen) => {
        setOpen(nextOpen);
        if (!nextOpen) {
          setIsCreating(false);
          setNewListName("");
          setSearchQuery("");
        }
      }}
    >
      <div className={cn("inline-block", className)}>
        <DropdownMenuTrigger asChild>{trigger}</DropdownMenuTrigger>
      </div>
      <DropdownMenuContent
        align={align}
        className="w-72 p-0"
        onCloseAutoFocus={(event) => {
          if (isCreating) event.preventDefault();
        }}
      >
        <div className="p-2 pb-3">
          <div
            className="w-full [&_input]:h-7 [&_input]:text-xs [&_input]:py-1 [&_svg]:size-3.5"
            onKeyDownCapture={(event) => {
              event.stopPropagation();
            }}
          >
            <SearchBar
              value={searchQuery}
              onChange={setSearchQuery}
              placeholder={hasSingleList ? "Search your list" : "Search"}
            />
          </div>
        </div>
        <DropdownMenuSeparator />
        <div
          className={cn("p-1", isListAreaScrollable && "overflow-y-auto")}
          style={
            isListAreaScrollable
              ? { maxHeight: `${MAX_VISIBLE_LISTS * 52}px` }
              : undefined
          }
        >
          {filteredLists.length > 0 ? (
            filteredLists.map((list) => {
              const inList = selected.has(list.id);
              return (
                <DropdownMenuCheckboxItem
                  key={list.id}
                  checked={inList}
                  onCheckedChange={() => toggleList(list.id)}
                  onSelect={(event) => event.preventDefault()}
                  className="py-2"
                >
                  <span className="min-w-0 leading-tight">
                    <span className="block truncate">{list.name}</span>
                    {typeof list.bookCount === "number" && (
                      <span className="block text-xs text-muted-foreground">
                        {list.bookCount}{" "}
                        {list.bookCount === 1 ? "book" : "books"}
                      </span>
                    )}
                  </span>
                </DropdownMenuCheckboxItem>
              );
            })
          ) : (
            <DropdownMenuItem disabled className="text-muted-foreground">
              {lists.length === 0 ? "No lists yet." : "No matching lists."}
            </DropdownMenuItem>
          )}
        </div>

        <DropdownMenuSeparator />
        <div className="p-1 pt-2">
          {isCreating ? (
            <div
              className="flex items-center gap-2"
              onKeyDown={(event) => {
                if (event.key === "Escape") {
                  event.preventDefault();
                  setIsCreating(false);
                }
              }}
            >
              <SearchBar
                inputRef={inputRef}
                showIcon={false}
                value={newListName}
                onChange={setNewListName}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    createList();
                  }
                }}
                placeholder="List name"
                className="flex-1"
                inputClassName="h-8 py-1 text-sm"
              />
              <Button
                type="button"
                size="sm"
                variant="secondary"
                onClick={createList}
              >
                Add
              </Button>
            </div>
          ) : (
            <DropdownMenuItem
              onSelect={(event) => {
                event.preventDefault();
                setIsCreating(true);
              }}
            >
              <Plus className="size-4 text-muted-foreground" />
              New list
            </DropdownMenuItem>
          )}
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
